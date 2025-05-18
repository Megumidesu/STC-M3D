import torch

class TriSimilarity:
    def __init__(self):
        pass

    def forward(self, candidates, query_reps):
        query_product = torch.ones_like(query_reps[0])
        for r in query_reps:
            query_product *= r.to(query_product.device)

        similarity_scores = query_product @ torch.t(candidates)

        if similarity_scores.dim() == 1:
            similarity_scores = similarity_scores.unsqueeze(0)

        assert similarity_scores.dim() == 2

        return similarity_scores

    def __call__(self, candidates, query_reps):
        return self.forward(candidates, query_reps)