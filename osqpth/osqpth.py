import torch
from torch.autograd import Function
import osqp
import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as sla
from .util import to_numpy


class OSQP(Function):
    def __init__(self,
                 P_idx, P_shape,
                 A_idx, A_shape,
                 eps_rel=1e-05,
                 eps_abs=1e-05,
                 verbose=False,
                 max_iter=10000):
        self.eps_abs = eps_abs
        self.eps_rel = eps_rel
        self.verbose = verbose
        self.max_iter = max_iter
        self.P_idx, self.P_shape = P_idx, P_shape
        self.A_idx, self.A_shape = A_idx, A_shape

        # TODO: Perform OSQP Setup first to allocate memory?

    def forward(self, P_val, q_val, A_val, l_val, u_val):
        """Solve a batch of QPs using OSQP.

        This function solves a batch of QPs, each optimizing over
        `n` variables and having `m` constraints.

        The optimization problem for each instance in the batch
        (dropping indexing from the notation) is of the form

            \hat x =   argmin_x 1/2 x' P x + q' x
                       subject to l <= Ax <= u

        where P \in S^{n,n},
              S^{n,n} is the set of all positive semi-definite matrices,
              q \in R^{n}
              A \in R^{m,n}
              l \in R^{m}
              u \in R^{m}

        These parameters should all be passed to this function as
        Variable- or Parameter-wrapped Tensors.
        (See torch.autograd.Variable and torch.nn.parameter.Parameter)

        If you want to solve a batch of QPs where `n` and `m`
        are the same, but some of the contents differ across the
        minibatch, you can pass in tensors in the standard way
        where the first dimension indicates the batch example.
        This can be done with some or all of the coefficients.

        You do not need to add an extra dimension to coefficients
        that will not change across all of the minibatch examples.
        This function is able to infer such cases.

        If you don't want to use any constraints, you can set the
        appropriate values to:

            e = Variable(torch.Tensor())

        """

        # Convert batches to sparse matrices/vectors
        self.n_batch = P_val.size(0) if len(P_val.size()) > 1 else 1
        self.m, self.n = self.A_shape   # Problem size
        Tensor = torch.cuda.DoubleTensor if P_val.is_cuda else torch.DoubleTensor

        # Convert P and A to sparse matrices
        # TODO (Bart): create CSC matrix during initialization. Then
        # just reassign the mat.data vector with A_val and P_val

        if self.n_batch == 1:
            # Create lists to make the code below work
            # TODO (Bart): Find a better way to do this
            P_val, q_val, A_val, l_val, u_val = [P_val], [q_val], [A_val], [l_val], [u_val]

        P = [spa.csc_matrix((to_numpy(P_val[i]), self.P_idx), shape=self.P_shape)
             for i in range(self.n_batch)]
        q = [to_numpy(q_val[i]) for i in range(self.n_batch)]
        A = [spa.csc_matrix((to_numpy(A_val[i]), self.A_idx), shape=self.A_shape)
             for i in range(self.n_batch)]
        l = [to_numpy(l_val[i]) for i in range(self.n_batch)]
        u = [to_numpy(u_val[i]) for i in range(self.n_batch)]

        # Perform forward step solving the QPs
        x_torch = Tensor().new_empty((self.n_batch, self.n), dtype=torch.double)

        x, y, z = [], [], []
        for i in range(self.n_batch):
            # Solve QP
            # TODO: Cache solver object in between
            m = osqp.OSQP()
            m.setup(P[i], q[i], A[i], l[i], u[i], verbose=self.verbose)
            result = m.solve()
            status = result.info.status
            if status != 'solved':
                # TODO: We can replace this with something calmer and
                # add some more options around potentially ignoring this.
                raise RuntimeError(f"Unable to solve QP, status: {status}")
            x.append(result.x)
            y.append(result.y)
            z.append(A[i].dot(result.x))
            x_torch[i] = Tensor(result.x)

        # Save stuff for backpropagation
        self.backward_vars = (P, q, A, l, u, x, y, z)

        # Return solutions
        return x_torch

    def backward(self, dl_dx_val):

        Tensor = torch.cuda.DoubleTensor if dl_dx_val.is_cuda else torch.DoubleTensor

        # Convert dl_dx to numpy
        dl_dx = to_numpy(dl_dx_val).squeeze()

        # Extract data from forward pass
        P, q, A, l, u, x, y, z = self.backward_vars

        # Convert to torch tensors
        nnz_P = len(self.P_idx[0])
        nnz_A = len(self.A_idx[0])        
        dP = Tensor().new_empty((self.n_batch, nnz_P), dtype=torch.double)
        dq = Tensor().new_empty((self.n_batch, self.n), dtype=torch.double)
        dA = Tensor().new_empty((self.n_batch, nnz_A), dtype=torch.double)
        dl = Tensor().new_empty((self.n_batch, self.m), dtype=torch.double)
        du = Tensor().new_empty((self.n_batch, self.m), dtype=torch.double)
        
        # TODO: Improve this, reuse OSQP, port it in C
        
        for i in range(self.n_batch):
            # Construct linear system
            # Taken from https://github.com/oxfordcontrol/osqp-python/blob/0363d028b2321017049360d2eb3c0cf206028c43/modulepurepy/_osqp.py#L1717
            # Guess which linear constraints are lower-active, upper-active, free
            ind_low = np.where(z[i] - l[i] < - y[i])[0]
            ind_upp = np.where(u[i] - z[i] < y[i])[0]
            n_low = len(ind_low)
            n_upp = len(ind_upp)

            # Form A_red from the assumed active constraints
            A_red = spa.vstack([A[i][ind_low], A[i][ind_upp]])

            # Form KKT linear system
            KKT = spa.vstack([spa.hstack([P[i], A_red.T]),
                              spa.hstack([A_red, spa.csc_matrix((n_low + n_upp, n_low + n_upp))])])
            rhs = np.hstack([dl_dx.squeeze(), np.zeros(n_low + n_upp)])

            # Get solution
            r_sol = sla.spsolve(KKT, rhs)
            r_x =  r_sol[:self.n]
            r_yl = r_sol[self.n:self.n + n_low]
            r_yu = r_sol[self.n + n_low:]
            r_y = np.zeros(self.m)
            r_y[ind_low] = r_yl
            r_y[ind_upp] = r_yu

             # Extract derivatives
            rows, cols = self.P_idx
            values = -.5 * (r_x[rows] * x[i][cols] + r_x[cols] * x[i][rows])
            dP[i] = Tensor(values)

            rows, cols = self.A_idx
            values = -(y[i][rows] * r_x[cols] + r_y[rows] * x[i][cols])
            dA[i] = Tensor(values)
            dq[i] = Tensor(-r_x)
            dl[i] = Tensor([r_yl[np.where(ind_low == j)[0]] if j in ind_low else 0
                            for j in range(self.m)])
            du[i] = Tensor([r_yu[np.where(ind_upp == j)[0]] if j in ind_upp else 0
                            for j in range(self.m)])

        grads = (dP, dq, dA, dl, du)

        return grads
