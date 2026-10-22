"""Terms of the linear product model and the spec that combines them.

A term declares one factor of the product without data: the raw column it reads, its basis and, optionally,
its ``reference``, where its factor is 1. ``ModelSpec`` combines terms, checks the combination without data
(``check``) and builds the fitting data once (``prepare``); a model fitted on that data keeps the prepared
spec and reads new rows with it.

A spec compiles to the blocks the solvers fit: each term is a feature group named after its input column
(a ``TensorTerm`` after its terms, ``a×b``), a ``ByTerm`` is the interaction of its term by its ``by`` term,
a ``FrozenTerm`` is a frozen submodel, and a term's reference becomes a ``fit(references=...)`` row.
"""
import copy
import itertools
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from quantbullet.preprocessing.transformers import TentTransformer
from .datacontainer import ProductModelDataContainer


class Term(ABC):
    """One factor of the product, declared without data; it becomes the feature group ``name``."""

    column: str
    reference = None

    @property
    def name(self):
        return self.column

    @property
    def inputs(self):
        """The raw columns the term reads."""
        return (self.column,)

    def has_reference(self):
        return self.reference is not None

    def setup_basis(self, frame, dtype):
        """Learn what needs data from the training ``frame`` and return its basis; ``prepare`` calls it on a copy."""
        return self.basis(frame, dtype)

    @abstractmethod
    def basis(self, frame, dtype):
        """Design columns of ``frame`` as ``dtype``, one per name in ``columns()``."""

    @abstractmethod
    def columns(self):
        """Names of the design columns, which name the coefficients too."""

    def reference_row(self):
        """Basis row of ``reference`` (the first of several): after ``fit`` the factor is 1 there."""
        return self.row(_references(self.reference)[0])

    def _tied(self, points, kind):
        """Indices among ``points`` of several references, which share one column; ``None`` for one or none."""
        references = _references(self.reference)
        if len(references) < 2:
            return None
        if len(set(references)) < len(references):
            raise ValueError(f"{self!r} repeats a reference: {references}")
        missing = [reference for reference in references if reference not in points]
        if missing:
            raise ValueError(f"{self!r}: several references must be {kind} {list(points)}; {missing} are not")
        return [points.index(reference) for reference in references]

    def row(self, value):
        """Basis row of one raw value."""
        return self.basis(pd.DataFrame({self.column: [value]}), np.float64)[0]

    def describe(self):
        """One line for reports: the basis, and the reference when there is one."""
        return self._detail() + ("" if self.reference is None else f", reference={self.reference!r}")

    def _detail(self):
        return ""

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class TentTerm(Term):
    """Tent basis of a numeric column (``TentTransformer``): the coefficients are the curve's values at the knots.

    ``reference`` is an x value, not necessarily a knot, or a list of knots. Several references are tied: their
    columns are summed into one, named ``{column}_at_{a}+{b}``, so the curve takes one value at all of them,
    which ``fit`` sets to the reference value. The first reference gauges the level; each further one is a
    constraint.
    """

    def __init__(self, column, knots, reference=None):
        self.column, self.knots, self.reference = column, knots, reference

    def setup_basis(self, frame, dtype):
        self.transformer_ = TentTransformer(self.knots)
        basis = self.transformer_.fit_transform(frame[[self.column]], dtype=dtype)
        self.tied_ = self._tied(self.transformer_.knots_.tolist(), "knots")
        return _tie(basis, self.tied_)

    def basis(self, frame, dtype):
        return _tie(self.transformer_.transform(frame[[self.column]], dtype=dtype), self.tied_)

    def columns(self):
        return _tied_names(self.transformer_.get_feature_names_out().tolist(), self.tied_, f"{self.column}_at_")

    def _detail(self):
        return f"knots={_numbers(self.knots)}"


class CategoricalTerm(Term):
    """One-hot levels of a column, named ``{column}_{level}``; the coefficients are the levels' values.

    ``levels`` defaults to the sorted levels in the training frame; missing values and other levels raise.
    ``reference`` is a level, or a list of levels tied into one column ``{column}_{a}+{b}`` (see ``TentTerm``).
    """

    def __init__(self, column, levels=None, reference=None):
        self.column, self.levels, self.reference = column, levels, reference

    def setup_basis(self, frame, dtype):
        values = frame[self.column]
        if values.isna().any():
            raise ValueError(f"{self.column} has missing values; map them to a level first")
        self.levels_ = (list(self.levels) if self.levels is not None
                        else np.unique(np.asarray(pd.unique(values), dtype=object)).tolist())
        self.tied_ = self._tied(self.levels_, "levels")
        return self.basis(frame, dtype)

    def basis(self, frame, dtype):
        values = frame[self.column]
        codes = pd.Index(self.levels_).get_indexer(values)
        if (codes < 0).any():
            raise ValueError(f"{self.column} has missing values or levels outside {self.levels_}: "
                             f"{pd.unique(values[codes < 0])[:5].tolist()}")
        column = _tied_positions(len(self.levels_), self.tied_)
        basis = np.zeros((len(codes), column.max() + 1), dtype=dtype)
        basis[np.arange(len(codes)), column[codes]] = 1
        return basis

    def columns(self):
        return _tied_names([f"{self.column}_{level}" for level in self.levels_], self.tied_, f"{self.column}_")

    def _detail(self):
        if hasattr(self, "levels_"):
            return f"{len(self.levels_)} levels"
        return "levels from data" if self.levels is None else f"levels={list(self.levels)}"


class BasisTerm(Term):
    """Any sklearn transformer of one column, such as ``FlatRampTransformer``; sparse output is made dense."""

    def __init__(self, column, transformer, reference=None):
        self.column, self.transformer, self.reference = column, transformer, reference

    def setup_basis(self, frame, dtype):
        return _dense(self.transformer.fit_transform(frame[[self.column]]), dtype)

    def basis(self, frame, dtype):
        return _dense(self.transformer.transform(frame[[self.column]]), dtype)

    def columns(self):
        return list(self.transformer.get_feature_names_out([self.column]))

    def _detail(self):
        knots = getattr(self.transformer, "knots", None)
        return type(self.transformer).__name__ + ("" if knots is None else f" knots={_numbers(knots)}")


class FrozenTerm(Term):
    """A fitted model whose ``predict`` of the column is the factor; it is not refitted and has no reference."""

    def __init__(self, column, model):
        self.column, self.model = column, model

    def basis(self, frame, dtype):
        return frame[[self.column]].to_numpy(dtype=dtype)

    def columns(self):
        return [self.column]

    def _detail(self):
        name = getattr(self.model, "model_name", type(self.model).__name__)
        return f"{name} {self.model.math_repr()}" if hasattr(self.model, "math_repr") else name


class ByTerm(Term):
    """``term`` with one curve per level of ``by``, a term of the same spec; each curve's level moves to ``by``.

    ``drop`` names the terms whose reference applies to the curves, ``"any"`` (both) or ``"all"`` (only the
    cell where both are at their references). It has no default: the model's other terms determine the one
    rule that removes exactly the duplicated parts, and ``ModelSpec.check`` accepts only that rule. The term
    is named after ``term`` and its reference is ``term``'s.
    """

    def __init__(self, term, by, drop=None):
        self.term, self.by, self.drop = term, by, drop

    @property
    def column(self):
        return self.term.column

    @property
    def reference(self):
        return self.term.reference

    def setup_basis(self, frame, dtype):
        return self.term.setup_basis(frame, dtype)

    def basis(self, frame, dtype):
        return self.term.basis(frame, dtype)

    def columns(self):
        return self.term.columns()

    def describe(self):
        return f"{self.term.describe()}; one curve per {self.by.name} level, drop={self.drop!r}"

    def __repr__(self):
        return f"ByTerm({self.term.name!r} by {self.by.name!r})"


class TensorTerm(Term):
    """The product of several terms' bases, one coefficient per cell of their knots and levels.

    ``terms`` are ``TentTerm`` or ``CategoricalTerm``, usually the spec's own main terms, so the knots and
    references are shared; a tent × tent term interpolates bilinearly on its node table. ``drop`` has no default
    (see ``ByTerm``): ``ModelSpec.check`` accepts only the rule the model's other terms determine. The cells it
    drops are fixed at the reference value: they are tied into one column, ``{name}:reference``, which ``fit``
    sets to that value. The term is named ``a×b`` unless ``name`` is given.

    Tent × tent cells are products of float32 values when prepared in float32, so their rows sum to 1 only
    within float32 rounding; prepare in float64 when exact reparametrization matters.
    """

    def __init__(self, *terms, drop=None, name=None):
        if len(terms) < 2:
            raise ValueError("TensorTerm takes at least two terms")
        self.terms, self.drop, self._name = list(terms), drop, name

    @property
    def name(self):
        return self._name or "×".join(term.name for term in self.terms)

    @property
    def inputs(self):
        return tuple(term.column for term in self.terms)

    def has_reference(self):
        return True

    def setup_basis(self, frame, dtype):
        """The terms are set up already (``ModelSpec.prepare`` sets each term up once)."""
        self.tied_ = self._dropped_cells()
        return self.basis(frame, dtype)

    def basis(self, frame, dtype):
        full = None
        for term in self.terms:
            basis = term.basis(frame, dtype)
            full = basis if full is None else (full[:, :, None] * basis[:, None, :]).reshape(len(basis), -1)
        return _tie(full, self.tied_)

    def columns(self):
        names = [""]
        for term in self.terms:
            names = [f"{a}×{b}" if a else b for a in names for b in term.columns()]
        tied = self.tied_
        return [f"{self.name}:reference" if j == tied[0] else name for j, name in enumerate(names) if j not in tied[1:]]

    def reference_row(self):
        row = np.zeros(len(self.columns()))
        row[self.tied_[0]] = 1.
        return row

    def points(self):
        """Each term's knots or levels, in column order."""
        return [term.transformer_.knots_.tolist() if isinstance(term, TentTerm) else list(term.levels_)
                for term in self.terms]

    def _dropped_cells(self):
        """Flat indices of the cells ``drop`` fixes: any dropped variable at its reference, or all at theirs."""
        variables = [term.name for term in self.terms]
        rule = _rule(self.drop, variables, self)
        points = self.points()
        positions = {term.name: point.index(_references(term.reference)[0])
                     for term, point in zip(self.terms, points) if term.reference is not None}
        cells = np.indices([len(point) for point in points]).reshape(len(points), -1).T
        if rule == "all":
            fixed = np.all([cells[:, k] == positions[v] for k, v in enumerate(variables)], axis=0)
        else:
            fixed = np.any([cells[:, k] == positions[v] for k, v in enumerate(variables) if v in rule], axis=0)
        return np.flatnonzero(fixed).tolist()

    def describe(self):
        return f"{' × '.join(term.name for term in self.terms)}, drop={self.drop!r}"

    def __repr__(self):
        return f"TensorTerm({self.name!r})"


def _rule(drop, variables, term):
    """``drop`` as ``"all"`` or the frozenset of names whose reference it drops."""
    if drop == "all":
        return "all"
    if drop == "any":
        return frozenset(variables)
    dropped = frozenset([drop] if isinstance(drop, str) else drop)
    unknown = sorted(dropped - set(variables))
    if unknown:
        raise ValueError(f"{term!r}: drop names {unknown}, which are not its terms {list(variables)}")
    return dropped


def _blocks(variables, rule):
    """The blocks a cross term over ``variables`` keeps under ``rule`` (note on tent bases, 4.2.1).

    A variable's columns span a constant and the part that varies with it, so the full term spans one block
    per subset of its variables. Dropping a variable's reference removes the blocks without that variable;
    ``"all"`` removes only the constant block.
    """
    subsets = {frozenset(c) for r in range(len(variables) + 1) for c in itertools.combinations(variables, r)}
    if rule == "all":
        return {block for block in subsets if block}
    return {block for block in subsets if rule <= block}


def _rule_text(rule, variables):
    if rule == "all":
        return "all"
    if len(rule) == len(variables):
        return "any"
    return tuple(v for v in variables if v in rule)


def _block_text(block, variables):
    if not block:
        return "the constant part"
    names = [v for v in variables if v in block]
    return f"the part that varies only with {names[0]}" if len(names) == 1 else \
        f"the part that varies with {' and '.join(names)} together"


def _cross_variables(term):
    """The variables (term names) of a cross term."""
    if isinstance(term, ByTerm):
        return (term.term.name, term.by.name)
    return tuple(component.name for component in term.terms)


def _references(reference):
    """``reference`` as a list: empty for none, one item for a single value."""
    if reference is None:
        return []
    return list(reference) if isinstance(reference, (list, tuple, np.ndarray)) else [reference]


def _tied_positions(n_columns, tied):
    """Column of each of ``n_columns`` points once the ``tied`` ones share the first tied point's column."""
    keep = [j for j in range(n_columns) if not tied or j not in tied[1:]]
    position = {j: i for i, j in enumerate(keep)}
    return np.array([position[tied[0]] if tied and j in tied else position[j] for j in range(n_columns)])


def _tie(basis, tied):
    """Sum the ``tied`` columns into the first of them and drop the others."""
    if not tied:
        return basis
    basis[:, tied[0]] = basis[:, tied].sum(axis=1)
    return np.delete(basis, tied[1:], axis=1)


def _tied_names(names, tied, prefix):
    """Column names once the ``tied`` columns are one, named ``prefix`` + their suffixes joined by ``+``."""
    if not tied:
        return names
    joined = prefix + "+".join(names[j][len(prefix):] for j in tied)
    return [joined if j == tied[0] else name for j, name in enumerate(names) if j not in tied[1:]]


def _numbers(values):
    return "[" + ", ".join(np.format_float_positional(float(value), trim="-") for value in values) + "]"


def _dense(basis, dtype):
    return np.asarray(basis.toarray() if hasattr(basis, "toarray") else basis, dtype=dtype)


def _values(frame, value):
    """A column of ``frame`` named by ``value``, or ``value`` itself as an array."""
    if value is None:
        return None
    return frame[value].to_numpy() if isinstance(value, str) else np.asarray(value)


class ModelSpec:
    """The terms of a model, checked together; ``prepare`` builds fitting data with a fitted copy of the spec."""

    def __init__(self, terms):
        self.terms = list(terms)

    def check(self):
        """Check the combination of terms without data; returns ``self``."""
        for term in self.terms:
            if not isinstance(term, Term):
                raise TypeError(f"ModelSpec takes terms; got {term!r}")
        names = [term.name for term in self.terms]
        shared = sorted({name for name in names if names.count(name) > 1})
        if shared:
            raise ValueError(f"Terms share the names {shared}; each term is one feature group named after its column")
        for term in self.terms:
            self._check_references(term)
            if isinstance(term, (ByTerm, TensorTerm)):
                self._check_cross(term)
        return self

    def _check_references(self, term):
        """Several references tie knots or levels, so only tent and categorical terms take them."""
        inner = term.term if isinstance(term, ByTerm) else term
        if len(_references(inner.reference)) < 2:
            return
        if not isinstance(inner, (TentTerm, CategoricalTerm)):
            raise ValueError(f"{term!r} has several references, which tie knots or levels; "
                             "only TentTerm and CategoricalTerm take them")
        if any(isinstance(other, ByTerm) and other.by is term for other in self.terms):
            raise ValueError(f"{term!r} is the by term of a ByTerm, which needs a column for each level; "
                             "give it one reference")
        if isinstance(inner, TentTerm):
            inner._tied(np.asarray(inner.knots, dtype=float).tolist(), "knots")

    def _provided_blocks(self, exclude):
        """``{block: [provider]}`` of every term but ``exclude``: the global level, each main term's varying part
        and each other cross term's kept blocks. Frozen terms provide nothing, since they cannot take what is dropped.
        """
        provided = {frozenset(): ["the global scalar"]}
        for term in self.terms:
            if term is exclude or isinstance(term, FrozenTerm):
                continue
            if isinstance(term, (ByTerm, TensorTerm)):
                variables = _cross_variables(term)
                if term.drop is None:
                    continue
                blocks = _blocks(variables, _rule(term.drop, variables, term))
            else:
                blocks = {frozenset([term.name])}
            for block in blocks:
                provided.setdefault(block, []).append(repr(term))
        return provided

    def _check_cross(self, term):
        """The one ``drop`` rule of a cross term (note on tent bases, 4.2): it keeps no block another term provides
        and removes no block that no other term provides. ``drop`` has no default, and any other rule raises.
        """
        if isinstance(term, ByTerm) and not any(other is term.by for other in self.terms):
            raise ValueError(f"{term!r}: its by term must be one of the spec's terms, the same object")
        variables = _cross_variables(term)
        if len(set(variables)) < len(variables):
            raise ValueError(f"{term!r} repeats a variable: {list(variables)}")
        provided = self._provided_blocks(exclude=term)
        full = _blocks(variables, frozenset())

        def problems(rule):
            kept = _blocks(variables, rule)
            return sorted(kept & set(provided), key=len), sorted(full - kept - set(provided), key=len)

        candidates = [frozenset(c) for r in range(len(variables), 0, -1)
                      for c in itertools.combinations(variables, r)] + ["all"]
        valid = [rule for rule in candidates if not any(problems(rule))]
        correct = (f"drop={_rule_text(valid[0], variables)!r}" if valid
                   else "no drop rule: removing exactly the repeated parts needs single cells")
        if term.drop is None:
            raise ValueError(f"{term!r} needs drop written out; for this model it is {correct}")
        rule = _rule(term.drop, variables, term)
        duplicated, missing = problems(rule)
        if duplicated or missing:
            reasons = [f"it keeps {_block_text(block, variables)}, which {' and '.join(provided[block])} also "
                       f"provides, so the model is collinear" for block in duplicated]
            reasons += [f"it removes {_block_text(block, variables)}, which no other term provides, so the model "
                        f"gains a constraint" for block in missing]
            raise ValueError(f"{term!r} has drop={term.drop!r}: {'; '.join(reasons)}. For this model it is {correct}")
        if isinstance(term, ByTerm):
            if rule != frozenset([term.term.name]):
                raise NotImplementedError(f"{term!r} fits a curve for every level of {term.by.name}, so it takes only "
                                          f"drop=({term.term.name!r},); use a TensorTerm for this model")
            return
        dropped = variables if rule == "all" else [v for v in variables if v in rule]
        main = {other.name: other for other in self.terms if not isinstance(other, (ByTerm, TensorTerm, FrozenTerm))}
        for component in term.terms:
            if not isinstance(component, (TentTerm, CategoricalTerm)):
                raise ValueError(f"{term!r} takes TentTerm and CategoricalTerm terms; {component!r} is neither")
            references = _references(component.reference)
            if len(references) > 1:
                raise ValueError(f"{term!r}: {component!r} has several references; a tensor term needs one per term")
            if component.name in dropped and not references:
                raise ValueError(f"{term!r} drops {component!r}'s reference, which it does not have")
            if isinstance(component, TentTerm):
                if references and float(references[0]) not in np.asarray(component.knots, dtype=float):
                    raise ValueError(f"{term!r}: {component!r}'s reference {references[0]!r} is not one of its knots")
                other = main.get(component.name)
                if isinstance(other, TentTerm) and other is not component and \
                        not set(np.asarray(component.knots, dtype=float)) <= set(np.asarray(other.knots, dtype=float)):
                    warnings.warn(f"{term!r}: {component!r} has knots that {other!r} lacks, so where its reference "
                                  "sits changes the fit (note on tent bases, 4.4); use the main term's knots or a "
                                  "subset", stacklevel=3)

    def prepare(self, frame, response=None, weights=None, dtype=np.float32):
        """Fitting data for ``frame``: a fitted copy of the spec learns levels and builds every basis once.

        ``response`` and ``weights`` are columns of ``frame`` or arrays. The spec itself does not change.
        """
        self.check()
        spec = copy.deepcopy(self)
        spec.dtype_ = np.dtype(dtype)
        done = set()

        def setup(term):
            """Each term object is set up once; a tensor term's terms are set up before it."""
            if id(term) in done:
                return term.basis(frame, spec.dtype_)
            for component in term.terms if isinstance(term, TensorTerm) else ():
                if id(component) not in done:
                    component.setup_basis(frame, spec.dtype_)
                    done.add(id(component))
            done.add(id(term))
            return term.setup_basis(frame, spec.dtype_)

        bases = [setup(term) for term in spec.terms]
        return spec._data(frame, bases, response, weights)

    def transform(self, frame, response=None, weights=None):
        """Data for new rows; the spec must be one fitted by ``prepare``, such as ``model.spec_``."""
        self._check_fitted("transform")
        return self._data(frame, [term.basis(frame, self.dtype_) for term in self.terms], response, weights)

    def reference_rows(self, values):
        """``{name: basis row}`` of ``{name: raw value}``, for ``fit(references=...)`` on a container.

        The spec must be one fitted by ``prepare``; a value need not be a knot.
        """
        self._check_fitted("reference_rows")
        terms = self.term_dict()
        unknown = sorted(set(values) - set(terms))
        if unknown:
            raise ValueError(f"No terms {unknown}; the terms are {list(terms)}")
        return {name: terms[name].row(value) for name, value in values.items()}

    def term_dict(self):
        return {term.name: term for term in self.terms}

    def _check_fitted(self, method):
        if not hasattr(self, "dtype_"):
            raise ValueError(f"{method} needs a spec fitted by prepare, such as model.spec_")

    def _data(self, frame, bases, response, weights):
        expanded = pd.concat([pd.DataFrame(basis, columns=term.columns(), index=frame.index)
                              for term, basis in zip(self.terms, bases)], axis=1)
        return ModelData(self, frame, expanded, response=_values(frame, response), weights=_values(frame, weights))

    def feature_groups(self):
        return {term.name: term.columns() for term in self.terms}

    def submodels(self):
        return {term.name: term.model for term in self.terms if isinstance(term, FrozenTerm)}

    def interactions(self):
        return {term.name: term.by.name for term in self.terms if isinstance(term, ByTerm)}

    def references(self):
        """``{group: basis row}`` of the terms with a reference, for ``fit(references=...)``."""
        return {term.name: term.reference_row() for term in self.terms if term.has_reference()}

    def __repr__(self):
        return f"ModelSpec({self.terms!r})"


class ModelData(ProductModelDataContainer):
    """Fitting data from ``ModelSpec.prepare``: the container the solvers read, with the fitted spec and weights."""

    def __init__(self, spec, orig_df, expanded_df, response=None, weights=None):
        super().__init__(orig_df, expanded_df, response=response, feature_groups=spec.feature_groups(),
                         groups_in_orig=False)
        self.spec, self.weights = spec, weights


class FittedTerm:
    """A term of a fitted model (``model.term(name)``): its coefficients and its factor at raw values."""

    def __init__(self, model, term):
        self.model, self.term = model, term

    @property
    def name(self):
        return self.term.name

    @property
    def coef(self):
        """Coefficients named by the term's columns; a ``ByTerm`` has one column per level of ``by``."""
        coef = self.model.coef_[self.name]
        if isinstance(self.term, ByTerm):
            return pd.DataFrame(dict(coef.categories), index=self.term.columns())
        return pd.Series(coef, index=self.term.columns(), name=self.name)

    def curve(self, values, level=None):
        """The factor at raw ``values``; a ``ByTerm`` needs the ``level`` of ``by`` whose curve to read.

        A ``TensorTerm`` takes ``values`` as a frame or dict of its input columns.
        """
        if isinstance(self.term, TensorTerm):
            return self.model.block_factor(self.name, self.term.basis(pd.DataFrame(values), np.float64))
        values = np.atleast_1d(np.asarray(values, dtype=object if isinstance(values, str) else None))
        basis = self.term.basis(pd.DataFrame({self.term.column: values}), np.float64)
        if not isinstance(self.term, ByTerm):
            if level is not None:
                raise ValueError(f"{self.term!r} has no levels; curve takes no level")
            return self.model.block_factor(self.name, basis)
        levels = list(self.model.coef_[self.name].categories)
        if level not in levels:
            raise ValueError(f"{self.term!r}: curve needs level= one of {levels}")
        return self.model.block_factor(self.name, basis, cat_series=pd.Series([level] * len(values)))

    def table(self):
        """Factors at the term's own points: a tent's knots or a categorical term's levels.

        A ``ByTerm`` gives one column per level of ``by``. Other terms have no such points; use ``curve``.
        """
        if isinstance(self.term, TensorTerm):
            if len(self.term.terms) != 2:
                raise ValueError(f"{self.term!r} has {len(self.term.terms)} terms; table() shows two, use curve")
            (rows, columns), (a, b) = self.term.points(), self.term.inputs
            grid = pd.DataFrame({a: np.repeat(rows, len(columns)), b: np.tile(np.asarray(columns, dtype=object), len(rows))})
            return pd.DataFrame(self.curve(grid).reshape(len(rows), len(columns)),
                                index=pd.Index(rows, name=a), columns=pd.Index(columns, name=b))
        inner = self.term.term if isinstance(self.term, ByTerm) else self.term
        if isinstance(inner, TentTerm):
            points = inner.transformer_.knots_.tolist()
        elif isinstance(inner, CategoricalTerm):
            points = list(inner.levels_)
        else:
            raise ValueError(f"{self.term!r} has no knots or levels to tabulate; use curve(values)")
        index = pd.Index(points, name=inner.column)
        if isinstance(self.term, ByTerm):
            return pd.DataFrame({level: self.curve(points, level) for level in self.model.coef_[self.name].categories},
                                index=index)
        return pd.Series(self.curve(points), index=index, name=self.name)

    def __repr__(self):
        return f"FittedTerm({self.term!r})"
