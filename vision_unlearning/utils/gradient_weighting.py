from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List
import torch
from libs.utils.logger import get_logger


logger = get_logger('gradient_weighting')


class GradientWeightingMethod(BaseModel, ABC):
    '''
    Method used to conciliate/harmonize/combine/weight the gradients of the different tasks

    Inspired by @article{navon2022multi,
        title={Multi-Task Learning as a Bargaining Game},
        author={Navon, Aviv and Shamsian, Aviv and Achituve, Idan and Maron, Haggai and Kawaguchi, Kenji and Chechik, Gal and Fetaya, Ethan},
        journal={arXiv preprint arXiv:2202.01017},
        year={2022}
    }
    Source: https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py
    '''

    @abstractmethod
    def weight_grads(self, grads_forget: List[torch.Tensor], grads_retain: List[torch.Tensor], accelerator) -> List[torch.Tensor]:
        '''
        @return scaled_grad
        '''
        pass


class GradientWeightingMethodNone(GradientWeightingMethod):
    '''
    No weighting is applied, takes just the forget gradients

    For debugging/comparison purposes
    '''
    def weight_grads(self, grads_forget: List[torch.Tensor], grads_retain: List[torch.Tensor], accelerator) -> torch.Tensor:
        return torch.cat([g.view(-1) for g in grads_forget])

    
class GradientWeightingMethodMunba(GradientWeightingMethod):
    '''
    Inspired by @misc{wu2025munbamachineunlearningnash,
        title={MUNBa: Machine Unlearning via Nash Bargaining}, 
        author={Jing Wu and Mehrtash Harandi},
        year={2025},
        eprint={2411.15537},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/pdf/2411.15537v1}, 
    }
    The closed-form solution is implemented as described in the V1 of the paper.
    '''
    def weight_grads(self, grads_forget: List[torch.Tensor], grads_retain: List[torch.Tensor], accelerator) -> torch.Tensor:
        # Stack gradients to form matrix G
        G = torch.stack([
            torch.cat([g.view(-1) for g in grads_retain]),
            torch.cat([g.view(-1) for g in grads_forget])
        ])
        K = G @ G.T  # Compute K = G^T G; It is a 2x2 tensor
        # Possible variation: K /= torch.norm(K)  # As recomended here: https://github.com/AvivNavon/nash-mtl/blob/main/methods/weight_methods.py#L231
        
        # Solve for α using narsh equation
        k11, k12, k22 = K[0, 0], K[0, 1], K[1, 1]
        alpha_retain = torch.sqrt((2 * k11 * k22 + k12 * torch.sqrt(k11 * k22)) / (k11**2 * k22 - k11 * k12**2))    # This is a Tensor of shape [], aka is a float
        alpha_forget = (1 - k11 * alpha_retain**2) / (k12 * alpha_retain)    
        alpha = torch.tensor([alpha_retain, alpha_forget]).reshape(2, 1)  # Typical values seem to be things like [0.0016, -0.0029]
        logger.debug(f"Alpha in this iteration: {alpha}")

        G = G.to(accelerator.device)
        alpha = alpha.to(accelerator.device)
        
        scaled_grad = G.T @ alpha
        # Possible variations:
        # scaled_grad /= 2*torch.abs(alpha).min()
        # scaled_grad /= 2*alpha.min()
        # scaled_grad /= torch.norm(alpha)
        # if <differentSign>: scaled_grad = G.T @ alpha; else: ...  # As recommended here: https://github.com/JingWu321/MUNBa/blob/d691e13885a373d97e4177cb051bd0dc64a9c732/SD/MUNBa_cls.py#L271

        return scaled_grad
