"""An additive design matrix from one-hot and tent encodings and their crosses, with every choice written out.

``Design`` writes a full-rank matrix ``Z`` for ``eta = Z @ beta``: a GLM, a log link or any linear model. It is
separate from the product model, whose ``ModelSpec`` builds blocks that its solvers fit one at a time.

The constant appears in ``Z`` once, carried by one term: the intercept (``intercept=True``), a main term with
``reference=None``, which keeps all its columns, or a cross that keeps its constant part. ``intercept`` has no
default. An encoding names its column and its ``reference``, the knot or level whose column the design drops, or
``None``; ``reference`` has no default. A ``Cross`` multiplies its encodings' bases, one column per cell of their
knots and levels; its ``drop`` lists the conditions under which the design drops a cell and has no default either.

A cross's cells split into parts by the columns that are off their reference at the cell. A condition, a column or
a tuple of columns all at their references, holds on every cell of the parts that vary with none of its columns, so
each effect the design already has on some of the cross's columns (the constant, a main term, a smaller cross)
gives one condition: the cross's other columns all at their references. A main term is a cross of one column: when
another term carries the constant, it gives the condition that drops the main term's reference column.
``Design.check`` accepts ``drop`` when it removes exactly the parts the other terms provide, and otherwise names the
conditions.
"""
import copy
import itertools
import warnings

import numpy as np
import pandas as pd

from .transformers import TentTransformer


class Encoding:
    """The basis of one column; ``reference`` is the knot or level whose column the design drops, or ``None``."""

    column: str
    reference = None

    @property
    def name(self):
        return self.column

    def reference_index(self):
        """Position of the reference among ``points()``; ``None`` for ``reference=None``."""
        return None if self.reference is None else self.points().index(self.reference)

    def dropped(self):
        """Which of ``columns()`` the design drops: the reference column, or none for ``reference=None``."""
        mask = np.zeros(len(self.points()), dtype=bool)
        if self.reference is not None:
            mask[self.reference_index()] = True
        return mask

    def __repr__(self):
        return f"{type(self).__name__}({self.column!r})"


class Tent(Encoding):
    """Tent basis of a numeric column (``TentTransformer``): one column ``{column}_at_{knot}`` per knot.

    The coefficients are the curve's values at the knots. ``reference`` is one of the knots, or ``None``.
    """

    def __init__(self, column, knots, *, reference):
        points = np.asarray(knots, dtype=float)
        if points.ndim != 1 or len(points) < 2 or not np.all(np.isfinite(points)) or not np.all(np.diff(points) > 0):
            raise ValueError(f"Tent({column!r}): knots must be at least two finite, strictly increasing values")
        if reference is not None and float(reference) not in points:
            raise ValueError(f"Tent({column!r}): reference {reference!r} is not one of the knots {points.tolist()}")
        self.column, self.knots, self.reference = column, tuple(knots), reference

    def fit(self, frame):
        """Check that every knot has data around it (``TentTransformer.fit``); returns ``self``."""
        self.transformer_ = TentTransformer(self.knots).fit(frame[[self.column]])
        return self

    def basis(self, frame, dtype):
        return self.transformer_.transform(frame[[self.column]], dtype=dtype)

    def columns(self):
        return self.transformer_.get_feature_names_out().tolist()

    def points(self):
        """The knots, in column order."""
        return np.asarray(self.knots, dtype=float).tolist()

    def reference_index(self):
        return None if self.reference is None else self.points().index(float(self.reference))


class OneHot(Encoding):
    """One column ``{column}_{level}`` per level; the coefficients are the levels' values.

    ``levels`` defaults to the sorted levels of the frame ``fit`` sees; missing values and other levels raise.
    ``reference`` is one of the levels, or ``None``.
    """

    def __init__(self, column, *, reference, levels=None):
        if levels is not None and reference is not None and reference not in list(levels):
            raise ValueError(f"OneHot({column!r}): reference {reference!r} is not one of the levels {list(levels)}")
        self.column, self.reference, self.levels = column, reference, levels

    def fit(self, frame):
        """Learn the levels and check the column has no others; returns ``self``."""
        values = frame[self.column]
        if values.isna().any():
            raise ValueError(f"{self.column} has missing values; map them to a level first")
        self.levels_ = (list(self.levels) if self.levels is not None
                        else np.unique(np.asarray(pd.unique(values), dtype=object)).tolist())
        if self.reference is not None and self.reference not in self.levels_:
            raise ValueError(f"{self!r}: reference {self.reference!r} is not one of the levels {self.levels_}")
        self._codes(values)
        return self

    def basis(self, frame, dtype):
        codes = self._codes(frame[self.column])
        basis = np.zeros((len(codes), len(self.levels_)), dtype=dtype)
        basis[np.arange(len(codes)), codes] = 1
        return basis

    def _codes(self, values):
        codes = pd.Index(self.levels_).get_indexer(values)
        if (codes < 0).any():
            raise ValueError(f"{self.column} has missing values or levels outside {self.levels_}: "
                             f"{pd.unique(values[codes < 0])[:5].tolist()}")
        return codes

    def columns(self):
        return [f"{self.column}_{level}" for level in self.levels_]

    def points(self):
        """The levels, in column order."""
        return self.levels_


class Cross:
    """The product of several encodings' bases: one column per cell of their knots and levels, named ``a_x_b``.

    Pass a main term's own object to share its knots and reference, or another object for the cross's own; its
    knots must then be the main term's or a subset of them. ``drop`` is a list of conditions, each a column or a
    tuple of columns all at their references, so the columns it names need references; the design drops the cells
    where any condition holds. ``Design.check`` accepts only the conditions the design's other terms determine.
    """

    def __init__(self, *encodings, drop):
        if len(encodings) < 2:
            raise ValueError("Cross takes at least two encodings")
        for encoding in encodings:
            if not isinstance(encoding, (Tent, OneHot)):
                raise TypeError(f"Cross takes Tent and OneHot encodings; got {encoding!r}")
        self.encodings, self.drop = list(encodings), drop
        if len(set(self.variables)) < len(self.variables):
            raise ValueError(f"{self!r} repeats a column")
        self.conditions = _conditions(drop, self.variables, self)
        named = set().union(*self.conditions)
        unanchored = [encoding.column for encoding in encodings if encoding.column in named and encoding.reference is None]
        if unanchored:
            raise ValueError(f"{self!r}: drop names {unanchored}, which have reference=None; a condition needs the "
                             "reference of each column it names")

    @property
    def name(self):
        return "_x_".join(self.variables)

    @property
    def variables(self):
        """The columns the cross reads, which ``drop`` names."""
        return tuple(encoding.column for encoding in self.encodings)

    def kept_parts(self):
        """The parts the cross keeps: each set of its columns that every condition meets."""
        return {part for part in _parts(self.variables) if all(part & condition for condition in self.conditions)}

    def columns(self):
        names = [""]
        for encoding in self.encodings:
            names = [f"{a}_x_{b}" if a else b for a in names for b in encoding.columns()]
        return names

    def dropped(self):
        """Which of ``columns()`` the design drops: the cells where any condition holds."""
        shape = [len(encoding.points()) for encoding in self.encodings]
        cells = np.indices(shape).reshape(len(shape), -1).T
        at_reference = {encoding.column: cells[:, k] == encoding.reference_index()
                        for k, encoding in enumerate(self.encodings) if encoding.reference is not None}
        mask = np.zeros(len(cells), dtype=bool)
        for condition in self.conditions:
            mask |= np.all([at_reference[variable] for variable in condition], axis=0)
        return mask

    @staticmethod
    def product(bases):
        """Row-wise tensor product of the encodings' bases, in ``columns()`` order."""
        full = bases[0]
        for basis in bases[1:]:
            full = (full[:, :, None] * basis[:, None, :]).reshape(len(basis), -1)
        return full

    def __repr__(self):
        return f"Cross({self.name!r})"


class Design:
    """Encodings and crosses checked together, and the full-rank matrix they make; ``fit`` learns on a copy.

    ``intercept`` says whether the first column is ``"intercept"``, the constant's carrier; it has no default.
    Each term is named after its column, a cross ``a_x_b`` after its columns; the names must differ, and none may
    be ``"intercept"``.
    """

    def __init__(self, terms, *, intercept):
        self.terms, self.intercept = list(terms), intercept

    def check(self):
        """Check the terms together without data; returns ``self``. ``fit`` calls it."""
        if not isinstance(self.intercept, bool):
            raise TypeError(f"intercept is True or False; got {self.intercept!r}")
        for term in self.terms:
            if not isinstance(term, (Tent, OneHot, Cross)):
                raise TypeError(f"Design takes Tent, OneHot and Cross terms; got {term!r}")
        names = [term.name for term in self.terms]
        shared = sorted({name for name in names if names.count(name) > 1} | {"intercept"} & set(names))
        if shared:
            raise ValueError(f"Terms share the names {shared}, or are named 'intercept'; each column has at most one "
                             "main term and each set of columns one cross")
        crosses = [frozenset(term.variables) for term in self.terms if isinstance(term, Cross)]
        if len(set(crosses)) < len(crosses):
            raise ValueError("Two crosses read the same columns; write one cross per set of columns")
        self._check_encodings()
        self._check_constant()
        for term in self.terms:
            if isinstance(term, Cross):
                self._check_cross(term)
        return self

    def _check_encodings(self):
        """The encodings of one column are of one kind; different references are legal, but warned about."""
        uses = {}
        for term in self.terms:
            for encoding in term.encodings if isinstance(term, Cross) else [term]:
                uses.setdefault(encoding.column, []).append((term, encoding))
        for column, pairs in uses.items():
            kinds = sorted({type(encoding).__name__ for _, encoding in pairs})
            if len(kinds) > 1:
                raise ValueError(f"{column} is encoded both as {' and as '.join(kinds)}; encode a column one way")
            references = {encoding.reference for _, encoding in pairs if encoding.reference is not None}
            if len(references) > 1:
                where = ", ".join(f"{encoding.reference!r} in {term!r}" for term, encoding in pairs
                                  if encoding.reference is not None)
                warnings.warn(f"{column} has different references ({where}). The design spans the same model as with "
                              f"one reference, but a cross's other columns then have their main effects read at its "
                              f"{column} reference; use one reference per column unless that is intended",
                              stacklevel=3)

    def _check_constant(self):
        """Exactly one term carries the constant."""
        carriers = (["the intercept"] if self.intercept else []) + [
            repr(term) for term in self.terms
            if (frozenset() in term.kept_parts() if isinstance(term, Cross) else term.reference is None)]
        if len(carriers) != 1:
            found = (f"is carried by {' and '.join(carriers)}, so the design is collinear" if carriers
                     else "is carried by no term, so the design gains a constraint")
            raise ValueError(f"The constant {found}. Exactly one term carries it: the intercept (intercept=True), a main "
                             "term with reference=None, or a cross whose drop keeps its constant part")

    def _provided(self, exclude):
        """``{part: [(provider, {column: encoding})]}`` of every term but ``exclude``: the intercept's constant, each
        main term's varying part (and the constant without a reference) and each other cross's kept parts."""
        provided = {frozenset(): [("the intercept", {})]} if self.intercept else {}
        for term in self.terms:
            if term is exclude:
                continue
            if isinstance(term, Cross):
                encodings = dict(zip(term.variables, term.encodings))
                for part in term.kept_parts():
                    provided.setdefault(part, []).append((repr(term), {v: encodings[v] for v in part}))
            else:
                for part in [frozenset([term.column])] + ([frozenset()] if term.reference is None else []):
                    provided.setdefault(part, []).append((repr(term), {v: term for v in part}))
        return provided

    def _check_cross(self, cross):
        """``drop`` removes exactly the parts the other terms provide, which use the cross's knots or finer ones."""
        variables = cross.variables
        parts = set(_parts(variables))
        provided = {part: providers for part, providers in self._provided(cross).items() if part in parts}
        if frozenset(variables) in provided:
            raise ValueError(f"{cross!r} adds nothing: {_part_text(frozenset(variables), variables)} is already "
                             f"provided by {_names(provided[frozenset(variables)])}")
        for part in reversed(_sorted(provided, variables)):
            for smaller in _sorted(parts - set(provided), variables):
                if smaller < part:
                    raise ValueError(
                        f"{cross!r}: no drop fits this design. {_part_text(part, variables)} is provided by "
                        f"{_names(provided[part])}, but no term provides {_part_text(smaller, variables)}, and a "
                        f"condition that removes the first removes the second too; give the second a term of its own, "
                        f"or leave the first to this cross")
        kept = cross.kept_parts()
        duplicated = _sorted(kept & set(provided), variables)
        missing = _sorted(parts - kept - set(provided), variables)
        if duplicated or missing:
            reasons = [f"it keeps {_part_text(part, variables)}, also provided by {_names(provided[part])}, so the "
                       f"design is collinear" for part in duplicated]
            reasons += [f"it removes {_part_text(part, variables)}, which no other term provides, so the design gains "
                        f"a constraint" for part in missing]
            raise ValueError(f"{cross!r} has drop={cross.drop!r}: {'; '.join(reasons)}.\n"
                             f"{_drop_table(variables, provided)}")
        own = dict(zip(variables, cross.encodings))
        for part in _sorted(parts - kept, variables):
            for provider, encodings in provided[part]:
                for variable in part:
                    if isinstance(own[variable], Tent) and \
                            not set(own[variable].points()) <= set(encodings[variable].points()):
                        raise ValueError(
                            f"{cross!r} leaves {_part_text(part, variables)} to {provider}, but its {variable} knots "
                            f"{own[variable].points()} are not a subset of {encodings[variable].points()}, so where "
                            f"its references sit would change the fit; use {provider}'s knots or a subset of them")

    def fit(self, frame):
        """Check the terms, then learn each encoding's levels and check its data on a copy; returns ``self``.

        An encoding object used by several terms is fitted once, so they share its levels and knots.
        """
        self.check()
        terms = copy.deepcopy(self.terms)
        levels = {}
        for encoding in _encodings(terms):
            encoding.fit(frame)
            if isinstance(encoding, OneHot) and levels.setdefault(encoding.column, encoding.levels_) != encoding.levels_:
                raise ValueError(f"{encoding.column} has different levels in different terms: "
                                 f"{levels[encoding.column]} and {encoding.levels_}")
        self.terms_ = terms
        return self

    def transform(self, frame, dtype=np.float64):
        """The design matrix of ``frame`` as a frame on its index, with the columns of ``groups()``."""
        groups = self.groups()
        bases = {}

        def basis(encoding):
            """Each encoding's basis once, however many terms use it."""
            if id(encoding) not in bases:
                bases[id(encoding)] = encoding.basis(frame, dtype)
            return bases[id(encoding)]

        values = np.empty((len(frame), sum(map(len, groups.values()))), dtype=dtype)
        start = 0
        if self.intercept:
            values[:, 0], start = 1, 1
        for term in self.terms_:
            full = Cross.product([basis(e) for e in term.encodings]) if isinstance(term, Cross) else basis(term)
            full = full[:, ~term.dropped()]
            values[:, start:start + full.shape[1]] = full
            start += full.shape[1]
        return pd.DataFrame(values, columns=[c for columns in groups.values() for c in columns], index=frame.index)

    def fit_transform(self, frame, dtype=np.float64):
        """``fit`` and ``transform`` on one frame; warns about columns with no data, whose coefficients are free."""
        design = self.fit(frame).transform(frame, dtype=dtype)
        empty = design.columns[~np.any(design.to_numpy() != 0, axis=0)].tolist()
        if empty:
            warnings.warn(f"{len(empty)} columns have no data in the frame, so their coefficients are not identified: "
                          f"{empty[:10]}", stacklevel=2)
        return design

    def groups(self):
        """``{term name: column names}`` of ``transform``, the intercept first when there is one."""
        if not hasattr(self, "terms_"):
            raise ValueError("The design is not fitted; call fit first")
        groups = {"intercept": ["intercept"]} if self.intercept else {}
        for term in self.terms_:
            groups[term.name] = [c for c, dropped in zip(term.columns(), term.dropped()) if not dropped]
        return groups

    def __repr__(self):
        return f"Design({self.terms!r}, intercept={self.intercept!r})"


def _conditions(drop, variables, cross):
    """``drop`` as a list of frozensets of columns."""
    if not isinstance(drop, list):
        raise TypeError(f"{cross!r}: drop is a list of conditions, such as ['age'] or [('age', 'incentive')]; "
                        f"got {drop!r}")
    conditions = []
    for condition in drop:
        names = (condition,) if isinstance(condition, str) else condition
        if not isinstance(names, tuple) or not names or not all(isinstance(name, str) for name in names):
            raise TypeError(f"{cross!r}: each condition in drop is a column or a tuple of columns; got {condition!r}")
        unknown = [name for name in names if name not in variables]
        if unknown:
            raise ValueError(f"{cross!r}: drop names {unknown}, which are not its columns {list(variables)}")
        conditions.append(frozenset(names))
    return conditions


def _encodings(terms):
    """Each encoding object once: the main terms and the crosses' encodings, in order."""
    seen, encodings = set(), []
    for term in terms:
        for encoding in term.encodings if isinstance(term, Cross) else [term]:
            if id(encoding) not in seen:
                seen.add(id(encoding))
                encodings.append(encoding)
    return encodings


def _parts(variables):
    return [frozenset(c) for r in range(len(variables) + 1) for c in itertools.combinations(variables, r)]


def _sorted(parts, variables):
    return sorted(parts, key=lambda part: (len(part), [variables.index(v) for v in variables if v in part]))


def _part_text(part, variables):
    if not part:
        return "the constant part"
    names = [v for v in variables if v in part]
    return f"the part that varies only with {names[0]}" if len(names) == 1 else \
        f"the part that varies with {' and '.join(names)} together"


def _condition(columns):
    return columns[0] if len(columns) == 1 else tuple(columns)


def _names(providers):
    return " and ".join(provider for provider, _ in providers)


def _drop_table(variables, provided):
    """Each effect on the cross's columns, the condition it gives, and the drop they make."""
    lines = [f"  {_names(provided[part])}: {_condition([v for v in variables if v not in part])!r}"
             for part in _sorted(provided, variables)]
    largest = [part for part in provided if not any(part < other for other in provided)]
    drop = [_condition([v for v in variables if v in condition])
            for condition in _sorted([frozenset(variables) - part for part in largest], variables)]
    return ("Each effect on its columns gives one condition, the cross's other columns all at their references:\n"
            + "\n".join(lines) + f"\nso for this design drop={drop!r}; conditions that others imply may be left out")
