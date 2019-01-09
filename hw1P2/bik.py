if(len(self.hide)==0):
            self.criterion(self.out1,labels)
            self.error=self.criterion.derivative()
            self.dW[0]=np.dot(self.x.T,self.error)/ len(self.x)
            self.db[0]= np.sum(self.error, axis=0, keepdims= True)/ len(self.x)
        elif(len(self.hide)>0):
            self.db=[]
            self.dW=[]
            #print("self.out1  shape ",np.array(self.out1).shape)
            #print("labels  shape ",np.array(labels).shape)
            self.error= self.criterion(self.out1,labels)
            #print("self.error  shape ",np.array(self.error).shape)
            self.z=self.criterion.derivative()

            #print("Z shape ",np.array(self.z).shape)
            self.fromAct=self.activations[len(self.hide)].derivative()*self.z
            #print("self.fromAct  shape ",np.array(self.fromAct).shape)

            for i in range(len(self.hide)):
                self.fromAct=self.activations[len(self.hide)].derivative()*self.z
                for j in range(len(self.hide)-1,i,-1):
                    self.out3=self.fromAct.dot(self.W[j].T)
                    print("self.W[j] shape ",np.array(self.W[j]).shape)
                    self.fromAct=self.activations[j-1].derivative()
                    print("self.fromAct hassii  shape ",np.array(self.fromAct).shape)
                print("self.outAll[i] hassii  shape ",np.array(self.outAll[i]).shape)
                self.dW.append(self.outAll[i].T.dot(self.fromAct)/len(labels))
                self.db.append(np.sum(self.fromAct,0)/len(labels))
            return self.error






            #self.dW[1]=np.dot(self.outAll[0].T,self.fromAct)/ len(self.x)
            #print(np.array(self.dW[1]).shape)
            #self.db[1]= np.sum(self.error1, axis=0, keepdims= True)/ len(self.x)


            #self.da0=np.dot(self.error1,self.W[1].T)
            #self.error0=np.dot(self.activations[0].derivative(),self.da0.T)
            #self.dW[0]=np.dot(self.x.T,self.error0)/ len(self.x)
            #self.db[0]= np.sum(self.error0, axis=0, keepdims= True)/ len(self.x)
            #for i in reversed(range(len(hide)))

            #return self.error1
                # backward propgate through the network