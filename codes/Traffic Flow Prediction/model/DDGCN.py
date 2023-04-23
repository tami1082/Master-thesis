import torch
import torch.nn.functional as F
import torch.nn as nn

class DGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim):
        super(DGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
    def forward(self, x, node_embeddings):
        #x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        #output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        # supports_t = supports.T
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # support_set_t = [torch.eye(node_num).to(supports.device), supports_t]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
            # support_set_t.append(torch.matmul(2 * supports_t, support_set_t[-1]) - support_set_t[-2])
        # support_set.append(support_set_t)
        supports = torch.stack(support_set, dim=0)
        # supports_t = torch.stack(support_set_t, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      #B, cheb_k, N, dim_in
        # x_g_t = torch.einsum("knm,bmc->bknc", supports_t, x)
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_g_t = x_g_t.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        # x_g = x_g + x_g_t
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return x_gconv
    

class embed_DGCN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(embed_DGCN, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = DGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = DGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    


class DDGCN(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DDGCN, self).__init__()
        assert num_layers >= 1, 'act as time spans.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.linear_p = nn.Linear(dim_in, dim_out)
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(embed_DGCN(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(embed_DGCN(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        layer_res = []
        layer_res.append(self.linear_p(x))
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings[t, :, :])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            layer_res.append(current_inputs)
            current_inputs = current_inputs + layer_res[i]
            # current_inputs = current_inputs + self.linear_p(x)
        output_hidden = torch.stack(output_hidden, dim=0)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden, layer_res

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)


class DDGCN_e(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(DDGCN_e, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.linear_p = nn.Linear(dim_in, dim_out)
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(embed_DGCN(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(embed_DGCN(node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, layer_res, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings[t, :, :])
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
            # layer_res.append(current_inputs)
            current_inputs = current_inputs + layer_res[i+1]
            # current_inputs = current_inputs + self.linear_p(x)
        output_hidden = torch.stack(output_hidden, dim=0)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

class comb_ed(nn.Module):
    def __init__(self, args):
        super(comb_ed, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.time_step = args.time_step
        self.num_layers = args.num_layers
        self.device = args.device

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(args.time_step,args.num_nodes, args.embed_dim), requires_grad=True)

        self.encoder = DDGCN(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)
        self.decoder = DDGCN_e(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)

        #predictor
        # self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)
        self.step_project = nn.Linear(args.time_step, args.horizon)
        self.project_layer = nn.Linear(args.rnn_units, args.output_dim)

    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        #encoder
        init_state = self.encoder.init_hidden(source.shape[0])
        _, output_hidden, layer_res = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden

        # decoder
        go_symbol = torch.zeros(source.shape).to(self.device)
        decoder_hidden_state = output_hidden
        decoder_input = go_symbol
        output_decoder, _ = self.decoder(decoder_input, decoder_hidden_state, layer_res, self.node_embeddings)

        #linear predictor
        time_project = self.step_project(output_decoder.permute(0, 2, 3, 1))
        output_decoder = self.project_layer(time_project.permute(0, 3, 1, 2))
        
        return output_decoder
    
