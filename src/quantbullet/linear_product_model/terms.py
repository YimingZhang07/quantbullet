"""Terms of the linear product model and the spec that combines them.

A term declares one factor of the product without data: the raw column it reads, its basis and, optionally,
its ``reference``, where its factor is 1. ``ModelSpec`` combines terms, checks the combination without data
(``check``) and builds the fitting data once (``prepare``); a model fitted on that data keeps the prepared
spec and reads new rows with it.

A spec compiles to the blocks the solvers fit: each term is a feature group named after its input column,
a ``ByTerm`` is the interaction of its term by its ``by`` term, a ``FrozenTerm`` is a frozen submodel, and
a term's reference becomes a ``fit(references=...)`` row.
"""
import copy
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
        """Basis row of ``reference``: after ``fit`` the factor is 1 there."""
        return self.basis(pd.DataFrame({self.column: [self.reference]}), np.float64)[0]

    def __repr__(self):
        return f"{type(self).__name__}({self.name!r})"


class TentTerm(Term):
    """Tent basis of a numeric column (``TentTransformer``): the coefficients are the curve's values at the knots.

    ``reference`` is an x value, not necessarily a knot.
    """

    def __init__(self, column, knots, reference=None):
        self.column, self.knots, self.reference = column, knots, reference

    def setup_basis(self, frame, dtype):
        self.transformer_ = TentTransformer(self.knots)
        return self.transformer_.fit_transform(frame[[self.column]]).astype(dtype, copy=False)

    def basis(self, frame, dtype):
        return self.transformer_.transform(frame[[self.column]]).astype(dtype, copy=False)

    def columns(self):
        return self.transformer_.get_feature_names_out().tolist()


class CategoricalTerm(Term):
    """One-hot levels of a column, named ``{column}_{level}``; the coefficients are the levels' values.

    ``levels`` defaults to the sorted levels in the training frame; missing values and other levels raise.
    ``reference`` is a level.
    """

    def __init__(self, column, levels=None, reference=None):
        self.column, self.levels, self.reference = column, levels, reference

    def setup_basis(self, frame, dtype):
        values = frame[self.column]
        if values.isna().any():
            raise ValueError(f"{self.column} has missing values; map them to a level first")
        self.levels_ = (list(self.levels) if self.levels is not None
                        else np.unique(np.asarray(pd.unique(values), dtype=object)).tolist())
        return self.basis(frame, dtype)

    def basis(self, frame, dtype):
        values = frame[self.column]
        codes = pd.Index(self.levels_).get_indexer(values)
        if (codes < 0).any():
            raise ValueError(f"{self.column} has missing values or levels outside {self.levels_}: "
                             f"{pd.unique(values[codes < 0])[:5].tolist()}")
        basis = np.zeros((len(codes), len(self.levels_)), dtype=dtype)
        basis[np.arange(len(codes)), codes] = 1
        return basis

    def columns(self):
        return [f"{self.column}_{level}" for level in self.levels_]


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


class FrozenTerm(Term):
    """A fitted model whose ``predict`` of the column is the factor; it is not refitted and has no reference."""

    def __init__(self, column, model):
        self.column, self.model = column, model

    def basis(self, frame, dtype):
        return frame[[self.column]].to_numpy(dtype=dtype)

    def columns(self):
        return [self.column]


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

    def __repr__(self):
        return f"ByTerm({self.term.name!r} by {self.by.name!r})"


def _dropped(term, variables):
    """``term.drop`` as ``"all"`` or the set of names whose reference it drops."""
    if term.drop == "all":
        return "all"
    if term.drop == "any":
        return set(variables)
    dropped = {term.drop} if isinstance(term.drop, str) else set(term.drop)
    unknown = sorted(dropped - set(variables))
    if unknown:
        raise ValueError(f"{term!r}: drop names {unknown}, which are not its terms {list(variables)}")
    return dropped


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
            if isinstance(term.reference, (list, tuple, np.ndarray)):
                raise NotImplementedError(f"{term!r} has several references; one reference per term is supported")
            if isinstance(term, ByTerm):
                self._check_by(term)
        return self

    def _check_by(self, term):
        """The one ``drop`` rule of a ``ByTerm`` (note on tent bases, 4.2.3).

        A variable's reference must apply to the cross term exactly when the term left without that variable is
        in the model; frozen terms do not count, since they cannot take what is dropped.
        """
        if not any(other is term.by for other in self.terms):
            raise ValueError(f"{term!r}: its by term must be one of the spec's terms, the same object")
        variables = (term.term.name, term.by.name)
        rest = dict(zip(variables, variables[::-1]))
        others = {other.name for other in self.terms if other is not term and not isinstance(other, FrozenTerm)}
        required = {variable for variable in variables if rest[variable] in others}
        rule = "any" if len(required) == 2 else tuple(v for v in variables if v in required) or "all"
        if term.drop is None:
            raise ValueError(f"{term!r} needs drop written out; for this model it is drop={rule!r}")
        dropped = _dropped(term, variables)
        if dropped == (required or "all"):
            return
        problems, missing, extra = [], required, set()
        if dropped == "all":
            problems.append(f"drop='all' fixes only the cell where {' and '.join(variables)} are both at their references")
        else:
            missing, extra = required - dropped, dropped - required
            if not required and not dropped:
                problems.append("the term repeats the global level, which only drop='all' removes")
        problems += [f"keeping {v}'s reference leaves the part that varies only with {rest[v]} in both this term and "
                     f"{rest[v]}'s own term, so the model is collinear" for v in variables if v in missing]
        problems += [f"dropping {v}'s reference removes the part that varies only with {rest[v]}, which no other term "
                     f"provides, so the model gains a constraint" for v in variables if v in extra]
        raise ValueError(f"{term!r} has drop={term.drop!r}: {'; '.join(problems)}. For this model it is drop={rule!r}")

    def prepare(self, frame, response=None, weights=None, dtype=np.float32):
        """Fitting data for ``frame``: a fitted copy of the spec learns levels and builds every basis once.

        ``response`` and ``weights`` are columns of ``frame`` or arrays. The spec itself does not change.
        """
        self.check()
        spec = copy.deepcopy(self)
        spec.dtype_ = np.dtype(dtype)
        bases = [term.setup_basis(frame, spec.dtype_) for term in spec.terms]
        return spec._data(frame, bases, response, weights)

    def transform(self, frame, response=None, weights=None):
        """Data for new rows; the spec must be one fitted by ``prepare``, such as ``model.spec_``."""
        if not hasattr(self, "dtype_"):
            raise ValueError("transform needs a spec fitted by prepare, such as model.spec_")
        return self._data(frame, [term.basis(frame, self.dtype_) for term in self.terms], response, weights)

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
        return {term.name: term.reference_row() for term in self.terms if term.reference is not None}

    def __repr__(self):
        return f"ModelSpec({self.terms!r})"


class ModelData(ProductModelDataContainer):
    """Fitting data from ``ModelSpec.prepare``: the container the solvers read, with the fitted spec and weights."""

    def __init__(self, spec, orig_df, expanded_df, response=None, weights=None):
        super().__init__(orig_df, expanded_df, response=response, feature_groups=spec.feature_groups())
        self.spec, self.weights = spec, weights
