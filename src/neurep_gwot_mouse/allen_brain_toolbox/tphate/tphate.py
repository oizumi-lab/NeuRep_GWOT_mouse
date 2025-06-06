import warnings

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import tasklogger
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from . import base, graphs, mds, utils, vne
from .api import Graph

_logger = tasklogger.get_tasklogger("graphtools")


class TPHATE(BaseEstimator):
    def __init__(
        self,
        n_components=2,
        knn=5,
        decay=40,
        n_landmark=2000,
        t="auto",
        gamma=1,
        n_pca=100,
        mds_solver="sgd",
        knn_dist="euclidean",
        knn_max=None,
        mds_dist="euclidean",
        mds="metric",
        n_jobs=1,
        random_state=None,
        verbose=1,
        potential_method=None,
        alpha_decay=None,
        njobs=None,
        k=None,
        a=None,
        smooth_window=1,
        **kwargs,
    ):
        if k is not None:
            knn = k
        if a is not None:
            decay = a
        self.n_components = n_components
        self.decay = decay
        self.knn = knn
        self.t = t
        self.n_landmark = n_landmark
        self.mds = mds
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.knn_max = knn_max
        self.mds_dist = mds_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.kwargs = kwargs
        self.smooth_window = smooth_window

        self.graph = None
        self._diff_potential = None
        self.embedding = None
        self.X = None
        self.optimal_t = None
        self.autocorr_op = None
        self.diff_op = None
        self.dropoff = None
        self.a = None

        if (alpha_decay is True and decay is None) or (alpha_decay is False and decay is not None):
            warnings.warn(
                "alpha_decay is deprecated. Use `decay=None` to disable alpha decay in future.",
                FutureWarning,
            )
            if not alpha_decay:
                self.decay = None
                self.alpha_decay = None

        if njobs is not None:
            warnings.warn("njobs is deprecated. Please use n_jobs in future.", FutureWarning)
            n_jobs = njobs
        self.n_jobs = n_jobs
        if potential_method is not None:
            if potential_method == "log":
                gamma = 1
            elif potential_method == "sqrt":
                gamma = 0
            else:
                raise ValueError(
                    f"potential_method {potential_method} not recognized. Please use gamma between -1 and 1"
                )
            warnings.warn(
                "potential_method is deprecated. "
                f"Setting gamma to {gamma} to achieve"
                f" {potential_method} transformation.",
                FutureWarning,
            )
        elif gamma > 0.99 and gamma < 1:
            warnings.warn(
                "0.99 < gamma < 1 is numerically unstable. Setting gamma to 0.99",
                RuntimeWarning,
            )
            gamma = 0.99
        self.gamma = gamma
        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0
        self.verbose = verbose
        self._check_params()
        _logger.set_level(verbose)

    @property
    def phate_diffop(self):
        """phate_diffop :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The diffusion operator built from the phate graph
        """
        if self.graph is not None:
            if isinstance(self.graph, graphs.LandmarkGraph):
                phate_diffop = self.graph.landmark_op
            else:
                phate_diffop = self.graph.diff_op
            if sparse.issparse(phate_diffop):
                phate_diffop = phate_diffop.toarray()
            # self.phate_diffop = phate_diffop
            return phate_diffop
        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

    def set_diff_op(self):
        """diff_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The dual-diffusion tphate operator. Combination of the phate diffusion operator and the autocorrelation operator.
        """
        _logger.info("Combining PHATE operator and autocorr operator")
        if self.diff_op is None:
            if self.phate_diffop is not None and self.autocorr_op is not None:
                if np.sum(self.autocorr_op) != 0:
                    self.diff_op = np.matmul(self.phate_diffop.T, self.autocorr_op)
                else:
                    self.diff_op = self.phate_diffop
                    _logger.info("No autocorrelation measured; converging with PHATE")
            else:
                raise NotFittedError(
                    "This TPHATE instance is not fitted yet. Call "
                    "'fit' with appropriate arguments before "
                    "using this method."
                )
        return

    def set_autocorr_op(self):
        """autocorr_op :  array-like, shape=[n_samples, n_samples] or [n_landmark, n_landmark]
        The autocorrelation view (second view of tphate).
        """
        if self.autocorr_op is None:
            self.autocorr_op = self._generate_autocorrelation_view()
            return
        else:
            raise NotFittedError(
                "This PHATE instance is not fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

    @property
    def diff_potential(self):
        """Interpolates the PHATE potential to one entry per cell
        This is equivalent to calculating infinite-dimensional PHATE,
        or running PHATE without the MDS step.
        Returns
        -------
        diff_potential : ndarray, shape=[n_samples, min(n_landmark, n_samples)]
        """
        diff_potential = self._calculate_potential()
        if isinstance(self.graph, graphs.LandmarkGraph):
            diff_potential = self.graph.interpolate(diff_potential)
        return diff_potential

    def _check_params(self):
        """Check PHATE parameters
        This allows us to fail early - otherwise certain unacceptable
        parameter choices, such as mds='mmds', would only fail after
        minutes of runtime.
        Raises
        ------
        ValueError : unacceptable choice of parameters
        """

        utils.check_positive(n_components=self.n_components, knn=self.knn)
        utils.check_int(n_components=self.n_components, knn=self.knn, n_jobs=self.n_jobs)
        utils.check_between(-1, 1, gamma=self.gamma)
        utils.check_if_not(None, utils.check_positive, decay=self.decay)
        utils.check_if_not(
            None,
            utils.check_positive,
            utils.check_int,
            n_landmark=self.n_landmark,
            n_pca=self.n_pca,
            knn_max=self.knn_max,
        )
        utils.check_if_not("auto", utils.check_positive, utils.check_int, t=self.t)
        if not callable(self.knn_dist):
            utils.check_in(
                [
                    "euclidean",
                    "precomputed",
                    "cosine",
                    "correlation",
                    "cityblock",
                    "l1",
                    "l2",
                    "manhattan",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                    "precomputed_affinity",
                    "precomputed_distance",
                ],
                knn_dist=self.knn_dist,
            )
        if not callable(self.mds_dist):
            utils.check_in(
                [
                    "euclidean",
                    "cosine",
                    "correlation",
                    "braycurtis",
                    "canberra",
                    "chebyshev",
                    "cityblock",
                    "dice",
                    "hamming",
                    "jaccard",
                    "kulsinski",
                    "mahalanobis",
                    "matching",
                    "minkowski",
                    "rogerstanimoto",
                    "russellrao",
                    "seuclidean",
                    "sokalmichener",
                    "sokalsneath",
                    "sqeuclidean",
                    "yule",
                ],
                mds_dist=self.mds_dist,
            )
        utils.check_in(["classic", "metric", "nonmetric"], mds=self.mds)
        utils.check_in(["sgd", "smacof"], mds_solver=self.mds_solver)

    def _set_graph_params(self, **params):
        try:
            self.graph.set_params(**params)
        except AttributeError:
            # graph not defined
            pass

    def _reset_graph(self):
        self.graph = None
        self._reset_potential()

    def _reset_potential(self):
        self._diff_potential = None
        self._reset_embedding()

    def _reset_embedding(self):
        self.embedding = None

    def set_params(self, **params):
        """Set the parameters on this estimator.
        Any parameters not given as named arguments will be left at their
        current value.
        Parameters
        ----------
        n_components : int, optional, default: 2
            number of dimensions in which the data will be embedded
        knn : int, optional, default: 5
            number of nearest neighbors on which to build kernel
        decay : int, optional, default: 40
            sets decay rate of kernel tails.
            If None, alpha decaying kernel is not used
        n_landmark : int, optional, default: 2000
            number of landmarks to use in fast PHATE
            landmarking is not suggested with TPHATE as it reduces the number of samples
            and breaks temporal continuity
        t : int, optional, default: 'auto'
            power to which the diffusion operator is powered.
            This sets the level of diffusion. If 'auto', t is selected
            according to the knee point in the Von Neumann Entropy of
            the diffusion operator
        gamma : float, optional, default: 1
            Informational distance constant between -1 and 1.
            `gamma=1` gives the PHATE log potential, `gamma=0` gives
            a square root potential.
        n_pca : int, optional, default: 100
            Number of principal components to use for calculating
            neighborhoods. For extremely large datasets, using
            n_pca < 20 allows neighborhoods to be calculated in
            roughly log(n_samples) time.
        mds_solver : {'sgd', 'smacof'}, optional (default: 'sgd')
            which solver to use for metric MDS. SGD is substantially faster,
            but produces slightly less optimal results. Note that SMACOF was used
            for all figures in the PHATE paper.
        knn_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean', 'cosine', 'precomputed'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for building kNN graph. Custom distance
            functions of form `f(x, y) = d` are also accepted. If 'precomputed',
            `data` should be an n_samples x n_samples distance or
            affinity matrix. Distance matrices are assumed to have zeros
            down the diagonal, while affinity matrices are assumed to have
            non-zero values down the diagonal. This is detected automatically
            using `data[0,0]`. You can override this detection with
            `knn_dist='precomputed_distance'` or `knn_dist='precomputed_affinity'`.
        knn_max : int, optional, default: None
            Maximum number of neighbors for which alpha decaying kernel
            is computed for each point. For very large datasets, setting `knn_max`
            to a small multiple of `knn` can speed up computation significantly.
        mds_dist : string, optional, default: 'euclidean'
            recommended values: 'euclidean' and 'cosine'
            Any metric from `scipy.spatial.distance` can be used
            distance metric for MDS
        mds : string, optional, default: 'metric'
            choose from ['classic', 'metric', 'nonmetric'].
            Selects which MDS algorithm is used for dimensionality reduction
        n_jobs : integer, optional, default: 1
            The number of jobs to use for the computation.
            If -1 all CPUs are used. If 1 is given, no parallel computing code
            is used at all, which is useful for debugging.
            For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used
        random_state : integer or numpy.RandomState, optional, default: None
            The generator used to initialize SMACOF (metric, nonmetric) MDS
            If an integer is given, it fixes the seed
            Defaults to the global `numpy` random number generator
        verbose : `int` or `boolean`, optional (default: 1)
            If `True` or `> 0`, print status messages
        k : Deprecated for `knn`
        a : Deprecated for `decay`
        Examples
        --------
        >>> import tphate
        >>> import matplotlib.pyplot as plt
        >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=50, n_branch=5,
        ...                                               branch_length=50)
        >>> tree_data.shape
        (250, 50)
        >>> tphate_operator = tphate.TPHATE(k=5, a=20, t=150)
        >>> tree_phate = tphate_operator.fit_transform(tree_data)
        >>> tree_phate.shape
        (250, 2)
        >>> tphate_operator.set_params(n_components=10)
        TPHATE(a=20, alpha_decay=None, k=5, knn_dist='euclidean', mds='metric',
           mds_dist='euclidean', n_components=10, n_jobs=1, n_landmark=2000,
           n_pca=100, njobs=None, potential_method='log', random_state=None, t=150,
           verbose=1)
        >>> tree_tphate = tphate_operator.transform()
        >>> tree_tphate.shape
        (250, 10)
        >>> # plt.scatter(tree_tphate[:,0], tree_tphate[:,1], c=tree_clusters)
        >>> # plt.show()
        Returns
        -------
        self
        """
        reset_kernel = False
        reset_potential = False
        reset_embedding = False

        # mds parameters
        if "n_components" in params and params["n_components"] != self.n_components:
            self.n_components = params["n_components"]
            reset_embedding = True
            del params["n_components"]
        if "mds" in params and params["mds"] != self.mds:
            self.mds = params["mds"]
            reset_embedding = True
            del params["mds"]
        if "mds_solver" in params and params["mds_solver"] != self.mds_solver:
            self.mds_solver = params["mds_solver"]
            reset_embedding = True
            del params["mds_solver"]
        if "mds_dist" in params and params["mds_dist"] != self.mds_dist:
            self.mds_dist = params["mds_dist"]
            reset_embedding = True
            del params["mds_dist"]

        # diff potential parameters
        if "t" in params and params["t"] != self.t:
            self.t = params["t"]
            reset_potential = True
            del params["t"]
        if "potential_method" in params:
            if params["potential_method"] == "log":
                params["gamma"] = 1
            elif params["potential_method"] == "sqrt":
                params["gamma"] = 0
            else:
                raise ValueError(
                    "potential_method {} not recognized. Please use gamma between -1 and 1".format(
                        params["potential_method"]
                    )
                )
            warnings.warn(
                "potential_method is deprecated. Setting gamma to {} to achieve {} transformation.".format(
                    params["gamma"], params["potential_method"]
                ),
                FutureWarning,
            )
            del params["potential_method"]
        if "gamma" in params and params["gamma"] != self.gamma:
            self.gamma = params["gamma"]
            reset_potential = True
            del params["gamma"]

        # kernel parameters
        if "k" in params and params["k"] != self.knn:
            self.knn = params["k"]
            reset_kernel = True
            del params["k"]
        if "a" in params and params["a"] != self.decay:
            self.decay = params["a"]
            reset_kernel = True
            del params["a"]
        if "knn" in params and params["knn"] != self.knn:
            self.knn = params["knn"]
            reset_kernel = True
            del params["knn"]
        if "knn_max" in params and params["knn_max"] != self.knn_max:
            self.knn_max = params["knn_max"]
            reset_kernel = True
            del params["knn_max"]
        if "decay" in params and params["decay"] != self.decay:
            self.decay = params["decay"]
            reset_kernel = True
            del params["decay"]
        if "n_pca" in params:
            if self.X is not None and params["n_pca"] >= np.min(self.X.shape):
                params["n_pca"] = None
            if params["n_pca"] != self.n_pca:
                self.n_pca = params["n_pca"]
                reset_kernel = True
                del params["n_pca"]
        if "knn_dist" in params and params["knn_dist"] != self.knn_dist:
            self.knn_dist = params["knn_dist"]
            reset_kernel = True
            del params["knn_dist"]
        if "n_landmark" in params and params["n_landmark"] != self.n_landmark:
            if self.n_landmark is None or params["n_landmark"] is None:
                # need a different type of graph, reset entirely
                self._reset_graph()
            else:
                self._set_graph_params(n_landmark=params["n_landmark"])
            self.n_landmark = params["n_landmark"]
            del params["n_landmark"]

        # parameters that don't change the embedding
        if "n_jobs" in params:
            self.n_jobs = params["n_jobs"]
            self._set_graph_params(n_jobs=params["n_jobs"])
            del params["n_jobs"]
        if "random_state" in params:
            self.random_state = params["random_state"]
            self._set_graph_params(random_state=params["random_state"])
            del params["random_state"]
        if "verbose" in params:
            self.verbose = params["verbose"]
            _logger.set_level(self.verbose)
            self._set_graph_params(verbose=params["verbose"])
            del params["verbose"]

        if reset_kernel:
            # can't reset the graph kernel without making a new graph
            self._reset_graph()
        if reset_potential:
            self._reset_potential()
        if reset_embedding:
            self._reset_embedding()

        self._set_graph_params(**params)

        self._check_params()
        return self

    def reset_mds(self, **kwargs):
        """
        Deprecated. Reset parameters related to multidimensional scaling
        Parameters
        ----------
        n_components : int, optional, default: None
            If given, sets number of dimensions in which the data
            will be embedded
        mds : string, optional, default: None
            choose from ['classic', 'metric', 'nonmetric']
            If given, sets which MDS algorithm is used for
            dimensionality reduction
        mds_dist : string, optional, default: None
            recommended values: 'euclidean' and 'cosine'
            Any metric from scipy.spatial.distance can be used
            If given, sets the distance metric for MDS
        """
        warnings.warn(
            "TPHATE.reset_mds is deprecated. Please use TPHATE.set_params in future.",
            FutureWarning,
        )
        self.set_params(**kwargs)

    def reset_potential(self, **kwargs):
        """
        Deprecated. Reset parameters related to the diffusion potential
        Parameters
        ----------
        t : int or 'auto', optional, default: None
            Power to which the diffusion operator is powered
            If given, sets the level of diffusion
        potential_method : string, optional, default: None
            choose from ['log', 'sqrt']
            If given, sets which transformation of the diffusional
            operator is used to compute the diffusion potential
        """
        warnings.warn(
            "TPHATE.reset_potential is deprecated. Please use TPHATE.set_params in future.",
            FutureWarning,
        )
        self.set_params(**kwargs)

    def _parse_input(self, X):
        # passing graphs to PHATE
        if isinstance(X, graphs.LandmarkGraph) or (isinstance(X, base.BaseGraph) and self.n_landmark is None):
            self.graph = X
            X = X.data
            n_pca = self.graph.n_pca
            update_graph = False
            if isinstance(self.graph, graphs.TraditionalGraph):
                precomputed = self.graph.precomputed
            else:
                precomputed = None
            return X, n_pca, precomputed, update_graph
        elif isinstance(X, base.BaseGraph):
            self.graph = None
            X = X.kernel
            precomputed = "affinity"
            n_pca = None
            update_graph = False
            return X, n_pca, precomputed, update_graph
        else:
            try:
                if isinstance(X, pygsp.graphs.Graph):
                    self.graph = None
                    X = X.W
                    precomputed = "adjacency"
                    update_graph = False
                    n_pca = None
                    return X, n_pca, precomputed, update_graph
            except NameError:
                # pygsp not installed
                pass

        # checks on regular data
        update_graph = True
        try:
            if isinstance(X, anndata.AnnData):
                X = X.X
        except NameError:
            # anndata not installed
            pass
        if not callable(self.knn_dist) and self.knn_dist.startswith("precomputed"):
            if self.knn_dist == "precomputed":
                # automatic detection
                if isinstance(X, sparse.coo_matrix):
                    X = X.tocsr()
                if X[0, 0] == 0:
                    precomputed = "distance"
                else:
                    precomputed = "affinity"
            elif self.knn_dist in ["precomputed_affinity", "precomputed_distance"]:
                precomputed = self.knn_dist.split("_")[1]
            else:
                raise ValueError(
                    "knn_dist {} not recognized. Did you mean "
                    "'precomputed_distance', "
                    "'precomputed_affinity', or 'precomputed' "
                    "(automatically detects distance or affinity)?"
                )
            n_pca = None
        else:
            precomputed = None
            if self.n_pca is None or self.n_pca >= np.min(X.shape):
                n_pca = None
            else:
                n_pca = self.n_pca
        return X, n_pca, precomputed, update_graph

    def _update_graph(self, X, precomputed, n_pca, n_landmark):
        if self.X is not None and not utils.matrix_is_equivalent(X, self.X):
            """
            If the same data is used, we can reuse existing kernel and
            diffusion matrices. Otherwise we have to recompute.
            """
            self._reset_graph()
        else:
            try:
                self.graph.set_params(
                    decay=self.decay,
                    knn=self.knn,
                    knn_max=self.knn_max,
                    distance=self.knn_dist,
                    precomputed=precomputed,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    n_pca=n_pca,
                    n_landmark=n_landmark,
                    random_state=self.random_state,
                )
                _logger.info("Using precomputed graph and diffusion operator...")
            except ValueError as e:
                # something changed that should have invalidated the graph
                _logger.debug(f"Reset graph due to {str(e)}")
                self._reset_graph()

    def fit(self, X):
        """Computes the diffusion operator
        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`. If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix
        Returns
        -------
        tphate_operator : TPHATE
        The estimator object
        """
        X, n_pca, precomputed, update_graph = self._parse_input(X)

        if precomputed is None:
            _logger.info(f"Running TPHATE on {X.shape[0]} observations and {X.shape[1]} variables.")
        else:
            _logger.info(f"Running TPHATE on precomputed {precomputed} matrix with {X.shape[0]} observations.")

        if self.n_landmark is None or X.shape[0] <= self.n_landmark:
            n_landmark = None
        else:
            _logger.info(f"Landmarking not recommended; setting n_landmark to {X.shape[0]}")
            self.n_landmark = X.shape[0]
            n_landmark = None

        if self.graph is not None and update_graph:
            self._update_graph(X, precomputed, n_pca, n_landmark)

        self.X = X

        if self.graph is None:
            with _logger.task("graph and diffusion operator"):
                self.graph = Graph(
                    X,
                    n_pca=n_pca,
                    n_landmark=n_landmark,
                    distance=self.knn_dist,
                    precomputed=precomputed,
                    knn=self.knn,
                    knn_max=self.knn_max,
                    decay=self.decay,
                    thresh=1e-4,
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    random_state=self.random_state,
                    smooth_window=self.smooth_window,
                    **(self.kwargs),
                )
        self.set_autocorr_op()
        self.set_diff_op()
        return self

    def transform(self, X=None, t_max=100, plot_optimal_t=False, ax=None):
        """Computes the position of the cells in the embedding space
        Parameters
        ----------
        X : array, optional, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Not required, since TPHATE does not currently embed
            cells not given in the input matrix to `TPHATE.fit()`.
            Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData`. If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix
        t_max : int, optional, default: 100
            maximum t to test if `t` is set to 'auto'
        plot_optimal_t : boolean, optional, default: False
            If true and `t` is set to 'auto', plot the Von Neumann
            entropy used to select t
        ax : matplotlib.axes.Axes, optional
            If given and `plot_optimal_t` is true, plot will be drawn
            on the given axis.
        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
        The cells embedded in a lower dimensional space using TPHATE
        """
        if self.graph is None:
            raise NotFittedError(
                "This TPHATE instance is not fitted yet. Call "
                "'fit' with appropriate arguments before "
                "using this method."
            )
        elif X is not None and not utils.matrix_is_equivalent(X, self.X):
            # fit to external data
            warnings.warn(
                "Pre-fit TPHATE should not be used to transform a "
                "new data matrix. Please fit TPHATE to the new"
                " data by running 'fit' with the new data.",
                RuntimeWarning,
            )
            if isinstance(self.graph, graphs.TraditionalGraph) and self.graph.precomputed is not None:
                raise ValueError("Cannot transform additional data using a precomputed distance matrix.")
            else:
                if self.embedding is None:
                    self.transform()
                transitions = self.graph.extend_to_data(X)
                return self.graph.interpolate(self.embedding, transitions)
        else:
            diff_potential = self._calculate_potential(t_max=t_max, plot_optimal_t=plot_optimal_t, ax=ax)
            if self.embedding is None:
                with _logger.task(f"{self.mds} MDS"):
                    self.embedding = mds.embed_MDS(
                        diff_potential,
                        ndim=self.n_components,
                        how=self.mds,
                        solver=self.mds_solver,
                        distance_metric=self.mds_dist,
                        n_jobs=self.n_jobs,
                        seed=self.random_state,
                        verbose=max(self.verbose - 1, 0),
                    )
            if isinstance(self.graph, graphs.LandmarkGraph):
                _logger.debug("Extending to original data...")
                return self.graph.interpolate(self.embedding)
            else:
                return self.embedding

    def fit_transform(self, X, **kwargs):
        """Computes the diffusion operator and the position of the cells in the
        embedding space
        Parameters
        ----------
        X : array, shape=[n_samples, n_features]
            input data with `n_samples` samples and `n_dimensions`
            dimensions. Accepted data types: `numpy.ndarray`,
            `scipy.sparse.spmatrix`, `pd.DataFrame`, `anndata.AnnData` If
            `knn_dist` is 'precomputed', `data` should be a n_samples x
            n_samples distance or affinity matrix
        kwargs : further arguments for `TPHATE.transform()`
            Keyword arguments as specified in :func:`~tphate.TPHATE.transform`
        Returns
        -------
        embedding : array, shape=[n_samples, n_dimensions]
            The cells embedded in a lower dimensional space using TPHATE
        """
        with _logger.task("TPHATE"):
            self.fit(X)
            embedding = self.transform(**kwargs)
        return embedding

    def _generate_autocorrelation_view(self, smooth_window=None):
        """
        Generates a diffusion operator based on the calculated autocorrelation function
        of the input data (as smoothed over smooth_window).

        smooth_window: int, used for convolving the autocorrelation function in a rolling avg

        Output:
        M : numpy array of [n_samples, n_samples] symmetric matrix of the autocorrelation at each lag, row normalized

        """

        _logger.info("Learning the autocorrelation function...")
        if smooth_window is None:
            smooth_window = self.smooth_window
        if smooth_window is None:
            raise ValueError("smooth_window not set")

        n_samples, n_features = self.X.shape
        # calculate and store the AC functions for each feature separately
        A_feat = np.empty((n_samples, n_features))
        for f in range(n_features):
            A_feat[:, f] = sm.tsa.acf(self.X[:, f], fft=False, nlags=n_samples - 1, missing="drop")
        A_feat = np.nanmean(A_feat, axis=1)  # average over features to get one function
        acf = np.convolve(A_feat, np.ones(smooth_window), "same") / smooth_window  # rolling average
        dropoff = np.where(acf < 0)[0][0]  # timepoint where rolling average drops off
        self.dropoff = dropoff
        with _logger.task("Autocorr kernel"):
            _logger.info(f"Dropoff point: {self.dropoff}")
        # Spread out the autocorr function
        M = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if 0 < abs(i - j) < dropoff:
                    M[i, j] = acf[abs(i - j)]
                    M[j, i] = acf[abs(i - j)]
        # row normalize to turn to probabilities
        for row in M:
            if np.sum(row) == 0:  # this should never be true
                continue
            row[:] /= np.sum(row)
        temp = M[0, 1]
        M[0, 1] = M[1, 0]
        M[1, 0] = temp
        return M

    def _calculate_potential(self, t=None, t_max=100, plot_optimal_t=False, ax=None):
        """Calculates the diffusion potential
        Parameters
        ----------
        t : int
            power to which the diffusion operator is powered
            sets the level of diffusion
        t_max : int, default: 100
            Maximum value of `t` to test
        plot_optimal_t : boolean, default: False
            If true, plots the Von Neumann Entropy and knee point
        ax : matplotlib.Axes, default: None
            If plot=True and ax is not None, plots the VNE on the given axis
            Otherwise, creates a new axis and displays the plot
        Returns
        -------
        diff_potential : array-like, shape=[n_samples, n_samples]
            The diffusion potential fit on the input data
        """
        if t is None:
            t = self.t
        if self._diff_potential is None:
            if t == "auto":
                t = self._find_optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)
            else:
                t = self.t

            with _logger.task("diffusion potential"):
                # diffused diffusion operator
                diff_op_t = np.linalg.matrix_power(self.diff_op, t)
                if self.gamma == 1:
                    diff_op_t = np.log(diff_op_t + 1e-7)
                    self._diff_potential = diff_op_t
                elif self.gamma == -1:
                    self._diff_potential = diff_op_t
                else:
                    c = (1 - self.gamma) / 2
                    self._diff_potential = ((diff_op_t) ** c) / c
        elif plot_optimal_t:
            self._find_optimal_t(t_max=t_max, plot=plot_optimal_t, ax=ax)

        self._diff_potential = np.nan_to_num(self._diff_potential, nan=0.0).T
        return self._diff_potential

    def _von_neumann_entropy(self, t_max=100):
        """Calculate Von Neumann Entropy
        Determines the Von Neumann entropy of the diffusion affinities
        at varying levels of `t`. The user should select a value of `t`
        around the "knee" of the entropy curve.
        We require that 'fit' stores the value of `TPHATE.diff_op`
        in order to calculate the Von Neumann entropy.
        Parameters
        ----------
        t_max : int, default: 100
            Maximum value of `t` to test
        Returns
        -------
        entropy : array, shape=[t_max]
            The entropy of the diffusion affinities for each value of `t`
        """
        t = np.arange(t_max)
        return t, vne.compute_von_neumann_entropy(self.diff_op, t_max=t_max)

    def _find_optimal_t(self, t_max=100, plot=False, ax=None):
        """Find the optimal value of t
        Selects the optimal value of t based on the knee point of the
        Von Neumann Entropy of the diffusion operator.
        Parameters
        ----------
        t_max : int, default: 100
            Maximum value of t to test
        plot : boolean, default: False
            If true, plots the Von Neumann Entropy and knee point
        ax : matplotlib.Axes, default: None
            If plot=True and ax is not None, plots the VNE on the given axis
            Otherwise, creates a new axis and displays the plot
        Returns
        -------
        t_opt : int
            The optimal value of t
        """
        with _logger.task("optimal t"):
            t, h = self._von_neumann_entropy(t_max=t_max)
            t_opt = vne.find_knee_point(y=h, x=t)
            _logger.info(f"Automatically selected t = {t_opt}")

        if plot:
            if ax is None:
                fig, ax = plt.subplots()
                show = True
            else:
                show = False
            ax.plot(t, h)
            ax.scatter(t_opt, h[t == t_opt], marker="*", c="k", s=50)
            ax.set_xlabel("t")
            ax.set_ylabel("Von Neumann Entropy")
            ax.set_title(f"Optimal t = {t_opt}")
            if show:
                plt.show()

        self.optimal_t = t_opt

        return t_opt
