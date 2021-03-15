class environment: #for no good reason, the environment is stored on cuda - slower for loops, but cba to constantly move it between cpu and gpu memory
    def __init__(self,minibatch_dim=MINIBATCH,npoints=NPOINTS):
        self.minibatch=minibatch_dim
        self.npoints=npoints
        self.n=npoints-1
        return
    
    def reset(self):
        self.points = torch.rand([self.minibatch,self.npoints-1,2],device="cuda")
        self.start = torch.rand([self.minibatch,2],device="cuda")
        self.finish, self.state = self.start, torch.cat((torch.repeat_interleave(self.start[:,None,:],self.npoints-1,axis=1),torch.repeat_interleave(self.start[:,None,:],self.npoints-1,axis=1),self.points),axis=2)
        self.n=self.npoints-1
        return
    
    def step(self, action):
        indices=[i for i in range(self.minibatch)]
        newstart = self.points[indices,action[indices],:]
        reward = -torch.sqrt(torch.sum((newstart-self.start)*(newstart-self.start),1)) #reward is negative, due to optimize code where policy net chooses action with maximum Q-value, i.e. maximum expected reward, i.e. shortest distance
        self.start = newstart
        self.n = self.n-1
        
        place_holder = torch.zeros([self.minibatch,self.n,2],device="cuda")
        
        for i in indices:
            j = action[i].item()
            place_holder[i,:,:] = torch.cat((self.points[i,:j,:],self.points[i,j+1:,:]))
        
        self.points = place_holder
        self.state = torch.cat((torch.repeat_interleave(self.start[:,None,:],self.n,axis=1),torch.repeat_interleave(self.finish[:,None,:],self.n,axis=1),self.points),axis=2)
        
        return reward