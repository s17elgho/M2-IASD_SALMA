def logistic(x):
    return np.log(1+np.exp(x))
def cost_logistic(A,y,x,lbda):
        yAx = y * A.dot(x)
        return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x) ** 2 / 2.
def compute_grad(A,y,x,lbda):
        yAx = y * A.dot(x)
        n = A.shape[0]
        aux = 1. / (1. + np.exp(yAx))
        return - (A.T).dot(y * aux) / n + lbda * x
def compute_hessian(A,y,x,lbda):
        H = np.zeros((d,d))
        for i in range(n):
            aux = np.exp(-y[i]*(A[i].dot(x)))
            a = A[i,:].reshape((1,5))
            b = A[i,:].reshape((5,1))

            G = (aux/(1+aux)**2)*(b.dot(a))
            for k in range(5):
                for l in range(5):
                    
                    H[k,l] += (1/n)*G[k,l]

                    
        return(H)
def gradient_descent(A,y,tau,niter,lbda):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    grad_values = []
    for i in range(niter):
        gd = compute_grad(A,y,x,lbda)
        grad_values.append(np.linalg.norm(gd))
        x = x-tau*gd
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values,grad_values
def nesterov_gradient_descent(A,y,tau,niter,lbda):
    d = A.shape[1]
    x_init = np.random.randn(d)
    z_init = np.random.randn(d)
    x = x_init
    z = z_init
    x_history = [x]
    cost_values = [cost_logistic(A,y,x,lbda)]
    lamda_nesterov = 0
    for i in range(1,niter):
        gd = compute_grad(A,y,z,lbda)
        x = z-tau*gd
        lamda_nesterov = 0.5*(1+np.sqrt(1+4*(lamda_nesterov**2)))
        lamda_nesterov_t1 = 0.5*(1+np.sqrt(1+4*(lamda_nesterov**2)))
        gamma_t = (lamda_nesterov-1)/lamda_nesterov_t1
        z = x + gamma_t*(x - x_history[-1])
        x_history.append(x)
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values
def grad_i(A, i, x, y, lbda):
        grad = - A[i] * y[i] / (1. + np.exp(y[i]* A[i].dot(x)))
        grad += lbda * x
        return grad    
def stoch_grad(A,tau,y,lbda,nb, niter, scaling = 0) :
    d = A.shape[1]
    n = A.shape[0]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    val_cost_values = []
    if scaling == 0:
        for k in range(niter): 
            # Draw the batch indices
            ik = np.random.choice(n,nb,replace=False)# Batch gradient
            # Stochastic gradient calculation
            sg = np.zeros(d)
            for j in range(nb):
                gi = grad_i(A,ik[j],x,y,lbda)
                sg = sg + gi
            sg = (1/nb)*sg
            x = x - tau * sg
            if ((k*nb) % n )== 0: 
                func = cost_logistic(A,y,x,lbda)
                cost_values.append(func)
                val_func = cost_logistic(A1,y1,x,lbda)
                val_cost_values.append(val_func)

            
    if scaling > 0:
        nu = 1/(2 *(n ** (0.5)))
        R = np.zeros(d)
        lamda = 0.8
        for k in range(niter): 
            # Draw the batch indices
            ik = np.random.choice(n,nb,replace=False)# Batch gradient
            # Stochastic gradient calculation
            sg = np.zeros(d)
            for j in range(nb):
                gi = grad_i(A,ik[j],x,y,lbda)
                sg = sg + gi
            sg = (1/nb)*sg
            
            if scaling==1:
            #tau = 0.
                R = lamda*R + (1-lamda)*sg*sg
            elif scaling==2:
                R = R + sg*sg 
            sg = sg/(np.sqrt(R+nu))
            x = x - tau * sg
            if ((k*nb) % n) == 0:  
                func = cost_logistic(A,y,x,lbda)
                cost_values.append(func)
                val_func = cost_logistic(A1,y1,x,lbda)
                val_cost_values.append(val_func)

    return x, cost_values, val_cost_values
def stoch_grad2(A,tau,y,lbda,nb, niter, scaling = 0) :
    d = A.shape[1]
    n = A.shape[0]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    val_cost_values = []
    if scaling == 0:
        for k in range(niter): 
            # Draw the batch indices
            ik = np.random.choice(n,nb,replace=False)# Batch gradient
            # Stochastic gradient calculation
            sg = np.zeros(d)
            for j in range(nb):
                gi = grad_i(A,ik[j],x,y,lbda)
                sg = sg + gi
            sg = (1/nb)*sg
            x = x - tau * sg
            func = cost_logistic(A,y,x,lbda)
            cost_values.append(func)
            val_func = cost_logistic(A1,y1,x,lbda)
            val_cost_values.append(val_func)

            
    if scaling > 0:
        nu = 1/(2 *(n ** (0.5)))
        R = np.zeros(d)
        lamda = 0.8
        for k in range(niter): 
            # Draw the batch indices
            ik = np.random.choice(n,nb,replace=False)# Batch gradient
            # Stochastic gradient calculation
            sg = np.zeros(d)
            for j in range(nb):
                gi = grad_i(A,ik[j],x,y,lbda)
                sg = sg + gi
            sg = (1/nb)*sg
            
            if scaling==1:
            #tau = 0.
                R = lamda*R + (1-lamda)*sg*sg
            elif scaling==2:
                R = R + sg*sg 
            sg = sg/(np.sqrt(R+nu))
            x = x - tau * sg 
            func = cost_logistic(A,y,x,lbda)
            cost_values.append(func)
            val_func = cost_logistic(A1,y1,x,lbda)
            val_cost_values.append(val_func)

    return x, cost_values, val_cost_values
def cost_logistic_lasso(A,y,x,lbda):
    yAx = y * A.dot(x)
    return np.mean(logistic(-yAx)) + lbda * np.linalg.norm(x,1)
def Soft(x,s): return np.maximum( abs(x)-s, np.zeros(x.shape)  ) * np.sign(x)
def ISTA(A,y,x,lbda,tau): 
    gd = compute_grad(A,y,x,lbda)
    return Soft( x-tau*(gd ), lbda*tau )

def forward_backward(A,y,lbda,tau,d):
    E_test = []
    flist = np.zeros((niter,1))
    x = np.zeros(d)
    for i in np.arange(0,niter):
        flist[i] = cost_logistic_lasso(A,y,x,lbda)
        x = ISTA(A,y,x,lbda,tau)
        yAx = y1 * A1.dot(x)
        E_test.append(np.mean(logistic(-yAx)))
    return(flist,x,E_test)

def Newton(A,y,niter,lbda):
    d = A.shape[1]
    x_init = np.random.randn(d)
    x = x_init
    cost_values = []
    for i in range(niter):
        gd = compute_grad(A,y,x,lbda)
        H = compute_hessian(A,y,x,lbda)
        x = x-np.linalg.pinv(H).dot(gd)
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
    return x,cost_values, H

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.0001, n_iters=100):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
# SVRG implementation
def svrg(A,y, n_iter=100,m=5): 
    objvals = []
    n = A.shape[0]
    L = np.linalg.norm(A, ord=2) ** 2 / (4. * n) 
    alpha = 0.2/L
    w0 = np.random.randn(A.shape[1])
    w = w0.copy()
    k=0
    obj = cost_logistic(A,y,w,lbda)
    objvals.append(obj);
    print("SVRG")
    while (k < n_iter):
        gwk = compute_grad(A,y,w,lbda)
        if (k+n)//n > k//n:
            objvals.append(obj)
        wtilda = w
        wtildavg = w
        for j in range(m):
            ij = np.random.choice(n,1,replace=True)
            
            sg = grad_i(A, ij[0], wtilda, y, lbda)-grad_i(A, ij[0], w, y, lbda)+gwk
            wtilda[:] = wtilda - alpha*sg 
            if (k+n+j)//n > (k+n)//n:
                objvals.append(obj)
        w[:] = wtilda.copy()
        obj = cost_logistic(A,y,w,lbda)
        k += 1
        if k+m+n % n == 0:
            objvals.append(obj)
    if k+m+n % n > 0:
        objvals.append(obj)
    w_output = w.copy()
    return w_output, np.array(objvals)
def grad_xi(A,y,i,xiw):
        x_i = A[i]
        return (xiw - y[i]) * x_i
def momentum(A,tau,y,lbda,niter,beta) :
    d = A.shape[1]
    n = A.shape[0]
    x_init = np.random.randn(d)
    x = x_init
    x_vals = [list(x)]
    x_vals = x_vals + [list(x)]
    cost_values = []
    val_cost_values = []

    for k in range(niter):
        # Draw the batch indices
        ik = np.random.choice(n,1,replace=False)# Batch gradient
        # Stochastic gradient calculation
        sg = np.zeros(d)
        old = np.array(x_vals[-1][0])

        gi = grad_i(A,ik,old,y,lbda)
        sg = gi
        very_old = np.array(x_vals[-2][0])
        sg = sg.reshape(d)
        x = old - tau * sg + beta*(old-very_old)
        
        x_vals.append([list(x)])
        func = cost_logistic(A,y,x,lbda)
        cost_values.append(func)
        val_func = cost_logistic(A1,y1,x,lbda)
        val_cost_values.append(val_func)

    return x, cost_values, val_cost_values
