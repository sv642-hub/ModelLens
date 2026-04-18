import torch


vocab = {0: "(", 1: ")", 2: "<pad>"}
vocab_size = len(vocab)
pad_id = 2


def generate_bracket_data(num_samples, seq_len):
        data = []
        for _ in range(num_samples):
            seq = []
            depth = 0
            for _ in range(seq_len):
                if depth == 0 or torch.rand(1) > 0.5:
                    seq.append(0)
                    depth +=1
                else:
                    seq.append(1)
                    depth -=1
            data.append(torch.tensor(seq))
        return torch.stack(data)