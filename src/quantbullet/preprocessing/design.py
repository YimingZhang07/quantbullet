"""An additive design matrix from one-hot and tent encodings and their crosses, with every choice written out.

``Design`` writes the matrix ``Z`` its terms describe, for ``eta = Z @ beta``: a GLM, a log link or any linear model.
It is separate from the product model, whose ``ModelSpec`` builds blocks that its solvers fit one at a time.

For ``Z`` to be full rank and span its terms' model, the constant appears once, carried by one term: the intercept
(``intercept=True``), a main term with ``reference=None``, which keeps all its columns, or a cross that keeps its
constant part. ``intercept`` has no default. An encoding names its column and its ``reference``, the knot or level
whose column the design drops, or ``None``; ``reference`` has no default. A ``Cross`` multiplies its encodings'
bases, one column per cell of their knots and levels; its ``drop`` lists the conditions under which the design drops
a cell and has no default either.

A cross's cells split into parts by the columns that are off their reference at the cell. A condition, a column or
a tuple of columns all at their references, holds on every cell of the parts that vary with none of its columns, so
each effect the design already has on some of the cross's columns (the constant, a main term, a smaller cross)
gives one condition: the cross's other columns all at their references. A main term is a cross of one column: when
another term carries the constant, it gives the condition that drops the main term's reference column.

``Design.check`` lists a ``Violation`` for each way the design departs from that: columns that are collinear, or a
part dropped that no other term provides, which constrains the model (a design without a constant, as when an offset
carries the level, is one). The design is still built as written; ``fit`` keeps the list on ``violations_`` and sums
it up in one ``DesignWarning``. Only terms that make no matrix raise.
"""
import copy
import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .transformers import TentTransformer


class Encoding:
    """The basis of one column; ``reference`` is the knot or level whose column the design drops, a list of them, or
    ``None``. The coefficients are read from the first reference; each further one drops one more column, which
    constrains the model (a curve held at 0 there too, or levels sharing the reference's value)."""

    column: str
    reference = None

    @property
    def name(self):
        return self.column

    def references(self):
        """The references in the order given, the first the one the coefficients are read from; ``[]`` for ``None``."""
        return _as_list(self.reference)

    def reference_indices(self):
        """Positions of the references among ``points()``."""
        points = self.points()
        return [points.index(reference) for reference in self.references()]

    def dropped(self):
        """Which of ``columns()`` the design drops: the references' columns, none for ``reference=None``."""
        mask = np.zeros(len(self.points()), dtype=bool)
        mask[self.reference_indices()] = True
        return mask

    def _check_references(self, points, what):
        """Each reference is one of ``points``, once, and at least one point is left."""
        references = self.references()
        for given, reference in zip(_as_list(self.reference), references):
            if reference not in points:
                raise ValueError(f"{self!r}: reference {given!r} is not one of the {what} {list(points)}")
        if len(set(references)) < len(references):
            raise ValueError(f"{self!r}: reference {self.reference!r} repeats a value")
        if references and set(references) >= set(points):
            raise ValueError(f"{self!r}: reference {self.reference!r} names every one of the {what}, so the term "
                             "would have no columns")

    def __repr__(self):
        return f"{type(self).__name__}({self.column!r})"


class Tent(Encoding):
    """Tent basis of a numeric column (``TentTransformer``): one column ``{column}_at_{knot}`` per knot.

    The coefficients are the curve's values at the knots. ``reference`` is one of the knots, a list of them, or
    ``None``.
    """

    def __init__(self, column, knots, *, reference):
        points = np.asarray(knots, dtype=float)
        if points.ndim != 1 or len(points) < 2 or not np.all(np.isfinite(points)) or not np.all(np.diff(points) > 0):
            raise ValueError(f"Tent({column!r}): knots must be at least two finite, strictly increasing values")
        self.column, self.knots, self.reference = column, tuple(knots), reference
        self._check_references(points.tolist(), "knots")

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

    def references(self):
        return [float(reference) for reference in _as_list(self.reference)]


class OneHot(Encoding):
    """One column ``{column}_{level}`` per level; the coefficients are the levels' values.

    ``levels`` defaults to the sorted levels of the frame ``fit`` sees; missing values and other levels raise.
    ``reference`` is one of the levels, a list of them, or ``None``. A cross of only some levels names the others:
    with ``reference=["C", "P"]``, only N has cells of its own, and P shares C's value.
    """

    def __init__(self, column, *, reference, levels=None):
        self.column, self.reference, self.levels = column, reference, levels
        if levels is not None:
            self._check_references(list(levels), "levels")

    def fit(self, frame):
        """Learn the levels and check the column has no others; returns ``self``."""
        values = frame[self.column]
        if values.isna().any():
            raise ValueError(f"{self.column} has missing values; map them to a level first")
        self.levels_ = (list(self.levels) if self.levels is not None
                        else np.unique(np.asarray(pd.unique(values), dtype=object)).tolist())
        self._check_references(self.levels_, "levels")
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
    tuple of columns all at their references (any of them, for a list), so the columns it names need references; the
    design drops the cells where any condition holds. ``Design.check`` accepts only the conditions the design's other
    terms determine.
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
        unanchored = [encoding.column for encoding in encodings if encoding.column in named and not encoding.references()]
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
        at_reference = {encoding.column: np.isin(cells[:, k], encoding.reference_indices())
                        for k, encoding in enumerate(self.encodings) if encoding.references()}
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


class DesignWarning(UserWarning):
    """A design built as written although it has violations; ``warnings.simplefilter("error", DesignWarning)`` makes
    them errors."""


@dataclass(frozen=True)
class Violation:
    """One way a design departs from a full-rank matrix that spans its terms' model; the design is built anyway.

    ``kind`` is ``"collinear"`` (columns repeat what other columns span), ``"constraint"`` (a part is dropped that no
    other term provides), ``"hierarchy"`` (no drop fits a cross, since a part below one that other terms provide is
    missing), ``"knots"`` (where a cross's references sit changes the fit), ``"references"`` (one column with
    different references: the same model, read differently), ``"encoding"`` (one column both a ``Tent`` and a
    ``OneHot``), ``"levels"`` (one column with different levels in different terms) or ``"no data"`` (columns with no
    data, whose coefficients are not identified). ``term`` names where it is: a term, a column or the constant.
    """
    kind: str
    term: str
    message: str

    def __str__(self):
        return self.message


class Design:
    """Encodings and crosses checked together, and the matrix they make; ``fit`` learns on a copy.

    ``intercept`` says whether the first column is ``"intercept"``, the constant's carrier; it has no default.
    Each term is named after its column, a cross ``a_x_b`` after its columns; the names must differ, and none may
    be ``"intercept"``.
    """

    def __init__(self, terms, *, intercept):
        self.terms, self.intercept = list(terms), intercept

    def check(self):
        """The design's violations, found without data: a list of ``Violation``, empty when the matrix is full rank and
        spans its terms' model. Raises only for terms that make no matrix. ``fit`` calls it."""
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
        violations = self._check_encodings() + self._check_further_references() + self._check_constant()
        for term in self.terms:
            if isinstance(term, Cross):
                violations += self._check_cross(term)
        return violations

    def _check_encodings(self):
        """The encodings of one column are of one kind and read their coefficients from one reference."""
        uses, violations = {}, []
        for term in self.terms:
            for encoding in term.encodings if isinstance(term, Cross) else [term]:
                uses.setdefault(encoding.column, []).append((term, encoding))
        for column, pairs in uses.items():
            kinds = sorted({type(encoding).__name__ for _, encoding in pairs})
            if len(kinds) > 1:
                violations.append(Violation("encoding", column, f"{column} is encoded both as "
                                            f"{' and as '.join(kinds)}; encode a column one way unless that is intended"))
            firsts = [(term, _as_list(encoding.reference)[0]) for term, encoding in pairs if encoding.references()]
            if len({first for _, first in firsts}) > 1:
                where = ", ".join(f"{first!r} in {term!r}" for term, first in firsts)
                violations.append(Violation(
                    "references", column, f"{column} has different references ({where}). The design spans the same "
                    f"model as with one reference, but a cross's other columns then have their main effects read at "
                    f"its {column} reference; use one reference per column unless that is intended"))
        return violations

    def _check_further_references(self):
        """Each reference after the first drops one more column of its term, a constraint."""
        violations = []
        for term in self.terms:
            for encoding in term.encodings if isinstance(term, Cross) else [term]:
                given = _as_list(encoding.reference)
                if len(given) > 1:
                    first, further = given[0], given[1:]
                    held = (f"is 0 at {further}, as at {first!r}" if isinstance(encoding, Tent)
                            else f"gives {further} the value of {first!r}")
                    violations.append(Violation(
                        "constraint", repr(term), f"{term!r} has the {encoding.column} references "
                        f"{encoding.reference!r}: its coefficients are read from {first!r}, and the others drop more "
                        f"columns, so the design gains a constraint: the term {held}"))
        return violations

    def _check_constant(self):
        """Exactly one term carries the constant."""
        carriers = (["the intercept"] if self.intercept else []) + [
            repr(term) for term in self.terms
            if (frozenset() in term.kept_parts() if isinstance(term, Cross) else not term.references())]
        if len(carriers) == 1:
            return []
        carriers_text = ("For a free constant, exactly one term carries it: the intercept (intercept=True), a main term "
                         "with reference=None, or a cross whose drop keeps its constant part")
        if carriers:
            return [Violation("collinear", "the constant", f"The constant is carried by {' and '.join(carriers)}, so "
                                                           f"the design is collinear. {carriers_text}")]
        return [Violation("constraint", "the constant", "The constant is carried by no term, so the design gains a "
                          "constraint: eta is 0 where every term is at its reference, as when an offset carries the "
                          f"level. {carriers_text}")]

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
                for part in [frozenset([term.column])] + ([] if term.references() else [frozenset()]):
                    provided.setdefault(part, []).append((repr(term), {v: term for v in part}))
        return provided

    def _check_cross(self, cross):
        """``drop`` removes exactly the parts the other terms provide, which use the cross's knots or finer ones."""
        variables = cross.variables
        parts = set(_parts(variables))
        provided = {part: providers for part, providers in self._provided(cross).items() if part in parts}
        if frozenset(variables) in provided:
            return [Violation("collinear", repr(cross), f"{cross!r} adds nothing: "
                              f"{_part_text(frozenset(variables), variables)} is already provided by "
                              f"{_names(provided[frozenset(variables)])}")]
        violations = []
        unfit = [(part, smaller) for part in reversed(_sorted(provided, variables))
                 for smaller in _sorted(parts - set(provided), variables) if smaller < part]
        if unfit:
            part, smaller = unfit[0]
            violations.append(Violation(
                "hierarchy", repr(cross),
                f"{cross!r}: no drop fits this design. {_part_text(part, variables)} is provided by "
                f"{_names(provided[part])}, but no term provides {_part_text(smaller, variables)}, and a condition "
                f"that removes the first removes the second too; give the second a term of its own, or leave the first "
                f"to this cross"))
        # The table ends with the drop that fits, so it is left out when none does.
        table = "" if violations else f"\n{_drop_table(variables, provided)}"
        kept = cross.kept_parts()
        duplicated = _sorted(kept & set(provided), variables)
        missing = _sorted(parts - kept - set(provided), variables)
        if duplicated:
            reasons = [f"it keeps {_part_text(part, variables)}, also provided by {_names(provided[part])}"
                       for part in duplicated]
            violations.append(Violation("collinear", repr(cross), f"{cross!r} has drop={cross.drop!r}: "
                                        f"{'; '.join(reasons)}, so the design is collinear.{table}"))
        if missing:
            reasons = [f"it removes {_part_text(part, variables)}, which no other term provides" for part in missing]
            violations.append(Violation("constraint", repr(cross), f"{cross!r} has drop={cross.drop!r}: "
                                        f"{'; '.join(reasons)}, so the design gains a constraint.{table}"))
        own = dict(zip(variables, cross.encodings))
        for part in _sorted((parts - kept) & set(provided), variables):
            for provider, encodings in provided[part]:
                for variable in part:
                    if isinstance(own[variable], Tent) and \
                            not set(own[variable].points()) <= set(encodings[variable].points()):
                        violations.append(Violation(
                            "knots", repr(cross),
                            f"{cross!r} leaves {_part_text(part, variables)} to {provider}, but its {variable} knots "
                            f"{own[variable].points()} are not a subset of {encodings[variable].points()}, so where "
                            f"its references sit changes the fit; use {provider}'s knots or a subset of them"))
        return violations

    def fit(self, frame):
        """Check the terms, then learn each encoding's levels and check its data on a copy; returns ``self``.

        The design's violations are on ``violations_``; when there are any, one ``DesignWarning`` sums them up. An
        encoding object used by several terms is fitted once, so they share its levels and knots.
        """
        self._fit(frame)
        _warn(self.violations_)
        return self

    def _fit(self, frame):
        violations = self.check()
        terms = copy.deepcopy(self.terms)
        levels = {}
        for encoding in _encodings(terms):
            encoding.fit(frame)
            if isinstance(encoding, OneHot) and levels.setdefault(encoding.column, encoding.levels_) != encoding.levels_:
                violations.append(Violation("levels", encoding.column, f"{encoding.column} has different levels in "
                                            f"different terms: {levels[encoding.column]} and {encoding.levels_}"))
        self.terms_, self.violations_ = terms, violations

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
        """``fit`` and ``transform`` on one frame; also records columns with no data, whose coefficients are free."""
        self._fit(frame)
        design = self.transform(frame, dtype=dtype)
        empty = design.columns[~np.any(design.to_numpy() != 0, axis=0)].tolist()
        if empty:
            terms = ", ".join(name for name, columns in self.groups().items() if set(columns) & set(empty))
            self.violations_.append(Violation("no data", terms, f"{len(empty)} columns have no data in the frame, so "
                                              f"their coefficients are not identified: {empty[:10]}"))
        _warn(self.violations_)
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


def _as_list(reference):
    """A reference as a list: ``[]`` for ``None``, the values of a list or tuple, else the one value."""
    if reference is None:
        return []
    return list(reference) if isinstance(reference, (list, tuple)) else [reference]


def _warn(violations):
    """One ``DesignWarning`` for all of a fit's violations, at the caller of ``fit`` or ``fit_transform``."""
    if violations:
        where = "; ".join(f"{violation.kind} in {violation.term}" for violation in violations)
        warnings.warn(f"The design has {len(violations)} violations and is built as written: {where}. Each is in full "
                      f"in design.violations_", DesignWarning, stacklevel=3)


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
