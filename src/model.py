import torch

class SiameseComparator(torch.nn.Module):
    def __init__(self, embeddings, emb_dim=64, docemb_dim=20, freeze_emb=True, nconv=32, seed=0):
        super(SiameseComparator, self).__init__()
        torch.manual_seed(seed)

        # ----- In ----
        if isinstance(embeddings, int):
            self.emb = torch.nn.Embedding(embeddings, emb_dim)
        else:
            emb_dim = embeddings.shape[1]
            self.emb = torch.nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=freeze_emb)

        # ----- conv ----
        self.conv2 = torch.nn.Conv1d(emb_dim, nconv, 2, padding='valid')
        self.conv3 = torch.nn.Conv1d(emb_dim, nconv, 3, padding='valid')
        self.conv4 = torch.nn.Conv1d(emb_dim, nconv, 4, padding='valid')

        # Input Layer
        self.fc_in = torch.nn.Linear(nconv * 3, docemb_dim)
        self.dout = torch.nn.Dropout(0.05)
        self.sig_in = torch.nn.Sigmoid()

        # Output layer
        self.fc_out = torch.nn.Linear(docemb_dim * 2, 1)  # Binary output. 1 when the first doc is better than second
        self.sig_out = torch.nn.Sigmoid()

    def forward(self, x1, x2):

        # Input 1
        emb1 = self.emb(x1).float()
        emb1 = emb1.transpose(1, 2).contiguous()

        cx1_2 = torch.nn.functional.relu(self.conv2(emb1))
        cx1_3 = torch.nn.functional.relu(self.conv3(emb1))
        cx1_4 = torch.nn.functional.relu(self.conv4(emb1))

        mcx1_2 = torch.nn.functional.max_pool1d(cx1_2, kernel_size=cx1_2.shape[2]).squeeze(dim=2)
        mcx1_3 = torch.nn.functional.max_pool1d(cx1_3, kernel_size=cx1_3.shape[2]).squeeze(dim=2)
        mcx1_4 = torch.nn.functional.max_pool1d(cx1_4, kernel_size=cx1_4.shape[2]).squeeze(dim=2)

        semb1 = torch.cat([mcx1_2, mcx1_3, mcx1_4], dim=1)

        x1 = self.dout(semb1)
        x1 = self.fc_in(x1)

        # Input 2
        emb2 = self.emb(x2).float()
        emb2 = emb2.transpose(1, 2).contiguous()

        cx2_2 = torch.nn.functional.relu(self.conv2(emb2))
        cx2_3 = torch.nn.functional.relu(self.conv3(emb2))
        cx2_4 = torch.nn.functional.relu(self.conv4(emb2))

        mcx2_2 = torch.nn.functional.max_pool1d(cx2_2, kernel_size=cx2_2.shape[2]).squeeze(dim=2)
        mcx2_3 = torch.nn.functional.max_pool1d(cx2_3, kernel_size=cx2_3.shape[2]).squeeze(dim=2)
        mcx2_4 = torch.nn.functional.max_pool1d(cx2_4, kernel_size=cx2_4.shape[2]).squeeze(dim=2)

        semb2 = torch.cat([mcx2_2, mcx2_3, mcx2_4], dim=1)

        x2 = self.dout(semb2)
        x2 = self.fc_in(x2)

        combined = torch.cat([x1, x2], dim=1)

        x_out = self.fc_out(combined)
        x_out = self.sig_out(x_out)

        return x_out