class ElasticNet:
    def __init__(self,training_input,real_output,test_percent,learning_rate=0.001):
        # basic info
        self.test_percent=test_percent
        self.train_percent=100-self.test_percent
        self.learning_rate=learning_rate
        self.lamb=0.003
        self.log=100
        self.alpha=0.5
        self.batch_size=4096
        # clean and converting
        self.data_cleaning_conversion(training_input,real_output)
        # spliting the data in train and test
        split = int(self.train_percent*len(self.training_input)/100)
        self.X_train=self.training_input[:split]
        self.Y_train=self.real_output[:split]
        self.X_test=self.training_input[split:]
        self.Y_test=self.real_output[split:]
        # normalization
        self.normalization()
        # shape of our data
        self.X_train_shape=self.X_train.shape
        self.Y_train_shape=self.Y_train.shape
        self.X_test_shape=self.X_test.shape
        self.Y_test_shape=self.Y_test.shape
        # adjusting our data
        self.X_train = np.hstack((self.X_train, np.ones((self.X_train.shape[0],1))))
        self.X_test = np.hstack((self.X_test, np.ones((self.X_test.shape[0],1))))
        # weight model
        weights=np.random.randn(self.X_train_shape[1],1)*0.003
        bias=np.random.rand(1,1)
        self.weights_bias = np.vstack((weights,bias))
        # history
        self.log_history=[]
        
    # methods
    def data_cleaning_conversion(self,training_input,real_output):
        training_input = np.nan_to_num(np.asarray(training_input))
        real_output = np.asarray(real_output).reshape(-1,1)
        self.training_input=training_input
        self.real_output=real_output
        # shuffling the data
        indices = np.random.permutation(len(self.training_input))
        self.training_input = self.training_input[indices]
        self.real_output = self.real_output[indices]

    def normalization(self):
        self.X_mean = np.mean(self.X_train, axis=0)
        self.X_std = np.std(self.X_train, axis=0)
        self.X_std[self.X_std == 0] = 1
        self.X_train = (self.X_train - self.X_mean) / self.X_std
        self.X_test = (self.X_test - self.X_mean) / self.X_std
        
        self.Y_mean = np.mean(self.Y_train)
        self.Y_std = np.std(self.Y_train)
        if self.Y_std == 0:
            self.Y_std = 1
        self.Y_train = (self.Y_train - self.Y_mean) / self.Y_std
        self.Y_test = (self.Y_test - self.Y_mean) / self.Y_std
        
    def prediction(self,X_batch):
        self.Y_pred=X_batch @ self.weights_bias
        
    def objective_function(self,y_batch):
        self.error=self.Y_pred-y_batch
        lasso = np.sum(np.abs(self.weights_bias[:-1]))
        ridge = np.sum(self.weights_bias[:-1]**2)
        self.mean_square_error = np.mean(self.error**2)
        self.total_error = np.mean(self.error**2) + self.lamb*( (self.alpha)*lasso + (1-self.alpha)*ridge)
        
    def parameter_updation_equation(self,X_batch,y_batch):
        y_pred = X_batch @ self.weights_bias
        self.error = y_pred - y_batch
        n = len(X_batch)
        gradient = (2/n) * (X_batch.T @ self.error) 
        w = self.weights_bias[:-1]
        l1_penalty = self.alpha * np.sign(w)
        l2_penalty = 2 * (1 - self.alpha) * w
        gradient[:-1] += self.lamb * (l1_penalty + l2_penalty)
        self.weights_bias -= self.learning_rate * gradient
        
    def batch_generate(self,X,y,batch_size):
        n=X.shape[0]
        for i in range(0,n,batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]
        

    # user call methods
    """
    implementing mini batch system
    """
    def training(self,epochs=1000):
        best_loss = float("inf")
        patience = 10
        counter = 0
        best_weights=np.zeros_like(self.weights_bias)
        for epoch in range(epochs):
            for X_batch, y_batch in self.batch_generate(self.X_train,self.Y_train,self.batch_size):
                # self.prediction(X_batch)
                # self.objective_function(y_batch)
                self.parameter_updation_equation(X_batch,y_batch)
            val_pred = self.X_test @ self.weights_bias
            val_loss = np.mean((val_pred - self.Y_test)**2)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights_bias.copy()
                counter = 0
            else:
                counter += 1
        
            if counter >= patience:
                self.weights_bias= best_weights
                return epoch
            if epoch > 0 and epoch % self.log == 0:
                self.learning_rate*=0.5
                # self.log_history.append({
                #     "epoch": epoch,
                #     "mse": float(self.mean_square_error),
                #     "total_error":float(self.total_error)
                # })
        self.weights_bias = best_weights
    def evaluate(self):
        y_pred_test = self.X_test @ self.weights_bias
        mse = np.mean((y_pred_test - self.Y_test)**2)
        return mse

    def predict(self,values):
        values = np.array(values)
        if values.ndim == 1:
            values = values.reshape(1,-1)
        values = (values - self.X_mean) / self.X_std
        values = np.hstack((values,np.ones((values.shape[0],1))))
        value=values @ self.weights_bias
        return  value*self.Y_std+self.Y_mean
    def r2_score(self):
        y_pred = self.X_test @ self.weights_bias
        ss_res = np.sum((self.Y_test - y_pred)**2)
        y_mean = np.mean(self.Y_test)
        ss_tot = np.sum((self.Y_test - y_mean)**2)
        if ss_tot == 0:
            return 0.0
        r2 = 1 - (ss_res / ss_tot)
        return r2
    def save_model(self,path="elastic_net_model.pkl"):
        import pickle
        with open(path, "wb") as f:
            self.X_train = None
            self.Y_train = None
            # self.X_test = None
            # self.Y_test = None
            self.log=None
            self.test_percent=None
            self.train_percent=None
            self.X_test_shape=None
            self.Y_test_shape=None
            self.X_train_shape=None
            self.Y_train_shape=None
            self.mean_square_error=None
            self.total_error=None
            self.error=None
            pickle.dump(self, f)
            return f"Saved at {path}"
    @staticmethod
    def load_model(path):
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
