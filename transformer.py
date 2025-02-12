import torch
import math

class MultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, W_q, W_k, W_v, W_o, num_heads):
        #batch = Number of examples/sequences in the batch.
        #seq_len: Length of each sequence (e.g., number of tokens).
        #d_model: Dimensionality of each token’s embedding (the feature size).
        batch, seq_len, d_model = query.size()
        #d_k is the dimensionality of each head, as total dimensionality is split to each head
        d_k = d_model // num_heads
        scale = 1.0 / math.sqrt(d_k)

        #Create the queries, keys and values matrices by multiplying by the learned weights
        Q = torch.einsum('btd,dh->bth', query, W_q)
        K = torch.einsum('btd,dh->bth', key, W_k)
        V = torch.einsum('btd,dh->bth', value, W_v)

        #Splitting the matrices into seperate heads 
        #Q/K/V are of shape (batch, seq_len, d_model), using .veiw(batch, seq_len, num_heads, d_k) splits the last dimension (d_model) into num_heads dimensions of size d_k (head dimensions)
        Q = Q.view(batch, seq_len, num_heads, d_k)
        K = K.view(batch, seq_len, num_heads, d_k)
        V = V.view(batch, seq_len, num_heads, d_k)
        
        #Swap the second dimension (originally seq_len) with the third dimension (originally num_heads) to get shape (batch,num_heads,seq_len,d_k)
        #Doing this allows us to compute dot products for each head individually much easier    
        Q = Q.transpose(1,2)
        V = V.transpose(1,2)
        K = K.transpose(1,2)

        #For each batch (b) and each head (n), for each query position (q), we perform a dot product between the query vector and the key vector
        #We sum over the dimension d, which is the per–head feature dimension.
        #Our output tenser here is of shape (batch, num_heads, seq_len, seq_len)
        #For each query position (q) --> 3rd dimension, we get a score for each key position (k) --> 4th dimension
        scores = torch.einsum('bnqd,bnkd->bnqk', Q, K)

        #Scale the scores by 1/sqrt(d_k) so they don't explode due to the summation
        scores = scores * scale

        #Apply softmax to the scores to get the attention weights
        #We want the attention weights for the key positions for each query to sum to 1
        attn = torch.nn.functional.softmax(scores, dim=-1)
        
        #For every batch and head, for each query position (q),sum the attention weights over all keys (indexed by k) * corresponding value vectors
        #One output vector per query for each head.
        head_outputs = torch.einsum('bnqk,bnkd->bnqd', attn, V)

        #Swap the second and third dimensions to get shape (batch, seq_len, num_heads, d_k)
        #Do this because we want to concatenate the heads together
        head_outputs = head_outputs.transpose(1, 2)

        #Concatenate the heads together using .view()
        #.contiguous() makes sure the memory is contiguous for efficiency
        concatenated = head_outputs.contiguous().view(batch, seq_len, d_model)

        #Multiply by the output weight matrix
        output = torch.einsum('btd,dh->bth', concatenated, W_o)

        #Save tensors for backwards pass
        ctx.save_for_backward(query, key, value, Q, K, V, attn, concatenated, W_q, W_k, W_v, W_o)
        ctx.num_heads = num_heads
        ctx.scale = scale
        return output


    @staticmethod
    def backward(ctx, grad_output):
        query, key, value, Q, K, V, attn, concat, W_q, W_k, W_v, W_o = ctx.saved_tensors
        num_heads = ctx.num_heads
        scale = ctx.scale
        batch, seq_len, d_model = query.size()
        d_k = d_model // num_heads

        # output = concatenated * W_o  (via einsum: 'btd,dh->bth')
        #So grad_concat =  
        grad_concat = torch.einsum('bth,hd->btd', grad_output, W_o.t())
        grad_W_o = torch.einsum('btd,bth->dh', concat, grad_output)

        ### Step B: Reshape grad_concat to get grad for head outputs ###
        # The concatenated tensor was formed by reshaping head_out (shape: (batch, num_heads, seq_len, d_k))
        grad_head_output = grad_concat.view(batch, seq_len, num_heads, d_k)
        grad_head_output = grad_head_output.transpose(1, 2)

        # Now grad_head_out has shape (batch, num_heads, seq_len, d_k)

        ### Step C: Gradients through the weighted sum: head_out = attn * V ###
        # For each element: H_{bnqd} = sum_{k} A_{bnqk} * V_{bnkd}
        grad_attn = torch.einsum('bnqd,bnkd->bnqk', grad_head_output, V)
        grad_V = torch.einsum('bnqk,bnqd->bnkd', attn, grad_head_output)

        ### Step D: Backprop through the softmax ###
        # For softmax, the derivative is:
        # dL/ds = A * (grad_attn - sum(grad_attn * A, axis=-1, keepdim=True))
        sum_grad = torch.sum(grad_attn * attn, dim=-1, keepdim=True)
        grad_scores  = attn * (grad_attn - sum_grad)
        # Account for the scaling in the forward pass:
        grad_scores = grad_scores * scale

        ### Step E: Backprop through the dot product: scores = einsum('bnqd,bnkd->bnqk', Q, K) * scale ###
        grad_Q = torch.einsum('bnqk,bnkd->bnqd', grad_scores, K)
        grad_K = torch.einsum('bnqk,bnqd->bnkd', grad_scores, Q)

        ### Step F: Reshape grad_Q, grad_K, grad_V back to (batch, seq_len, d_model) ###
        grad_Q_reshaped = grad_Q.transpose(1, 2)
        grad_Q_reshaped = grad_Q_reshaped.contiguous().view(batch, seq_len, d_model)

        grad_K_reshaped = grad_K.transpose(1, 2)
        grad_K_reshaped = grad_K_reshaped.contiguous().view(batch, seq_len, d_model)

        grad_V_reshaped = grad_V.transpose(1, 2)
        grad_V_reshaped = grad_V_reshaped.contiguous().view(batch, seq_len, d_model)

        ### Step G: Gradients for the initial linear projections ###
        grad_query = torch.einsum('btd,dh->bth', grad_Q_reshaped, W_q.t())
        grad_key   = torch.einsum('btd,dh->bth', grad_K_reshaped, W_k.t())
        grad_value = torch.einsum('btd,dh->bth', grad_V_reshaped, W_v.t())

        grad_W_q = torch.einsum('btd,bte->de', query, grad_Q_reshaped)
        grad_W_k = torch.einsum('btd,bte->de', key,   grad_K_reshaped)
        grad_W_v = torch.einsum('btd,bte->de', value, grad_V_reshaped)

        # The backward function returns a gradient for every input to the forward.
        # For non-tensor inputs (like num_heads), we return None.
        return grad_query, grad_key, grad_value, grad_W_q, grad_W_k, grad_W_v, grad_W_o, None


#Set up testing values

def custom_mha(query, key, value, W_q, W_k, W_v, W_o, num_heads):
    return MultiHeadAttention.apply(query, key, value, W_q, W_k, W_v, W_o, num_heads)

batch, seq_len, d_model, num_heads = 2, 5, 16, 4
standard_mha = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads).double()
query = torch.randn(batch, seq_len, d_model, dtype=torch.double, requires_grad=True)
key   = torch.randn(batch, seq_len, d_model, dtype=torch.double, requires_grad=True)
value = torch.randn(batch, seq_len, d_model, dtype=torch.double, requires_grad=True)

W_q = torch.randn(d_model, d_model, dtype=torch.double, requires_grad=True)
W_k = torch.randn(d_model, d_model, dtype=torch.double, requires_grad=True)
W_v = torch.randn(d_model, d_model, dtype=torch.double, requires_grad=True)
W_o = torch.randn(d_model, d_model, dtype=torch.double, requires_grad=True)

#Compare forward pass

with torch.no_grad():
    standard_mha.in_proj_weight.copy_(
        torch.cat((W_q.t(), W_k.t(), W_v.t()), dim=0))
    standard_mha.out_proj.weight.copy_(W_o.t())
    standard_mha.in_proj_bias.zero_()
    standard_mha.out_proj.bias.zero_()

query_std = query.transpose(0, 1).detach()
key_std   = key.transpose(0, 1).detach()
value_std = value.transpose(0, 1).detach()

standard_output, _ = standard_mha(query_std, key_std, value_std)
standard_output = standard_output.transpose(0, 1)

custom_output = custom_mha(query, key, value, W_q, W_k, W_v, W_o, num_heads)

print("Maximum absolute difference in forward outputs:", (custom_output - standard_output).abs().max().item())
#print("Difference in forward outputs:", (custom_output - standard_output))

#Compare backwards pass

inputs = (query, key, value, W_q, W_k, W_v, W_o, num_heads)

test = torch.autograd.gradcheck(custom_mha, inputs, eps=1e-6, atol=1e-4)
print("Gradient check passed?", test)