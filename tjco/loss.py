import torch
import torch.nn.functional as F

class TriadicLoss:
    def __init__(self, negative_sampling: str = "n"):
        self.negative_sampling = negative_sampling

    def compute_logits_n(self, anchor_rep, non_anchor_reps):
        non_anchor_shuff = torch.ones_like(anchor_rep)
        for r in non_anchor_reps:
            non_anchor_shuff = non_anchor_shuff * r[torch.randperm(r.shape[0])]

        logits = anchor_rep @ torch.t(non_anchor_shuff) # (bsz, bsz)

        MIP_of_positive_samples = anchor_rep.clone()
        for r in non_anchor_reps:
            MIP_of_positive_samples = MIP_of_positive_samples * r
        MIP_of_positive_samples = MIP_of_positive_samples.sum(axis=1) # (bsz)

        return torch.where(torch.eye(n=anchor_rep.shape[0]).to(anchor_rep.device) > 0.5,
                           MIP_of_positive_samples,
                           logits)

    def compute_non_anchor_products(self, tensors):
        if len(tensors) == 2:
            y, z = tensors
            y_z = []
            for i in range(y.shape[0]):
                y_z.append(y * z)
                z = torch.roll(z, shifts=1, dims=0)
            return y_z

        x = tensors[0]

        partial_products = self.compute_non_anchor_products(tensors[1:])

        all_products = []
        for i in range(x.shape[0]):
            for partial_product in partial_products:
                all_products.append(partial_product * x)
            x = torch.roll(x, shifts=1, dims=0)

        return all_products

    def compute_logits_n_squared(self, anchor_rep, non_anchor_reps):
        non_anchor_products = self.compute_non_anchor_products(non_anchor_reps)

        non_anchor_product = torch.cat(non_anchor_products, 0)

        logits = anchor_rep @ non_anchor_product.T

        return logits

    def forward(self, representations, logit_scale):
        labels = torch.arange(representations[0].shape[0]).to(representations[0].device)
        losses = []
        accuracy = []
        for i, r in enumerate(representations):
            if self.negative_sampling == "n":
                logits = logit_scale * self.compute_logits_n(r, [rep for j, rep in enumerate(representations) if i != j])
            elif self.negative_sampling == "n_squared":
                logits = logit_scale * self.compute_logits_n_squared(r, [rep for j, rep in enumerate(representations) if i != j])
            else:
                raise ValueError("Invalid value for negative_sampling. Expected 'n' or 'n_squared'.")

            loss = F.cross_entropy(logits, labels)

            losses.append(loss)
            acc = (logits.argmax(dim=1) == labels).float().mean()
            accuracy.append(acc)

        return sum(losses) / len(losses), sum(accuracy) / len(accuracy)

    def __call__(self, representations, logit_scale):
        return self.forward(representations, logit_scale)
    
def MMD(x, y, kernel):
    device = torch.device("cuda")
    eps = 1e-8  

    # Normalization
    x = (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0] + eps)
    y = (y - y.min(dim=0)[0]) / (y.max(dim=0)[0] - y.min(dim=0)[0] + eps)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":
        bandwidth_range = torch.tensor([0.2, 0.5, 0.9, 1.3]).to(device)
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx + eps)**-1
            YY += a**2 * (a**2 + dyy + eps)**-1
            XY += a**2 * (a**2 + dxy + eps)**-1

    if kernel == "rbf":
        bandwidth_range = torch.tensor([10, 15, 20, 50]).to(device)
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)