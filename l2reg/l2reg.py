import torch

class L2Reg:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, model1, model2):
        
        l2loss = 0
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            l2loss += self.alpha * torch.norm(p1 - p2)
            
        return l2loss

    def __repr__(self):
        return f"L2Reg({self.alpha})"

if __name__ == "__main__":
    print(f"L2 Regularization")