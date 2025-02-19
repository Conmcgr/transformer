import torch
import math

class MultiHeadAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, W_q, W_k, W_v, W_o, num_heads):
        #batch = Number of examples/sequences in the batch.
        #seq_len: How many tokens/patches are in each sequence --> If we do 2x2 patching on 28x28 mnist images we get 14*14 = 196 patches per sequence
        #d_model: Dimensionality of each token’s embedding (the feature size) -> Not necessarily the patch dimensionality usually we project onto higher dimensionality was slightly confused why?
        #I didn't implement the pre-attention embeding and assume that the input is already in the shape of (batch, seq_len, d_model)
        batch, seq_len, d_model = query.size()
        #d_k is the dimensionality of each head, as we will project down the total dimensionality to each head
        d_k = d_model // num_heads
        scale = 1.0 / math.sqrt(d_k)

        #Create the queries, keys and values matrices by multiplying by the learned weights
        #Here b = batch, t = seq_len, d = d_model, and h = output dimensionality which in most cases is also d_model
        Q = torch.einsum('btd,dh->bth', query, W_q)
        K = torch.einsum('btd,dh->bth', key, W_k)
        V = torch.einsum('btd,dh->bth', value, W_v)

        #Splitting the matrices into seperate heads 
        #Q/K/V are of shape (batch, seq_len, d_model)
        #.veiw(batch, seq_len, num_heads, d_k) is a command that splits the last dimension (d_model) into num_heads dimensions of size d_k (head dimensions)
        #For example if we had Q of shape (32, 196, 16), we could split it into 4 heads of size 4 by using .view(32, 196, 4, 4)
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
        #Our output tensor here is of shape (batch, num_heads, seq_len, seq_len)
        #For each query position (q) --> 3rd dimension, we get a score for each key position (k) --> 4th dimension
        #This produces a scalar value for each combination of a query position q and a key position k, for each head n and for each batch element b
        #Does so by computing a dot product between the query and key vectors for every single possible query and key position and summing for each
        scores = torch.einsum('bnqd,bnkd->bnqk', Q, K)

        #Scale the scores by 1/sqrt(d_k) so they don't explode due to the summation
        #This is done to prevent the dot product from becoming too large or too small
        #Now we can thunk of scores as a tensor represents the similarity between a specific query and a key for each batch and head element 
        scores = scores * scale

        #Apply softmax to the scores to get the attention weights
        #We want the attention weights for the key positions for each query to sum to 1
        attn = torch.nn.functional.softmax(scores, dim=-1)
        
        #For every batch and head, for each query position (q),sum the attention weights over all keys (indexed by k) * corresponding value vectors
        #For every example (batch index b) and every head (n), for each query position (q), you take the corresponding attention weights over all key positions (given by attn[b,n,q,k]) and use them to weight the corresponding value vectors (from V[b,n,k,d]).
        #The sum over k produces a new vector for each query position, which is a weighted sum of the value vectors (value weighted by query).
        #Aggregates information from every token in the input for each query, resulting in a new representation for each token that takes into account the entire input sequence.
        head_outputs = torch.einsum('bnqk,bnkd->bnqd', attn, V)

        #Swap the second and third dimensions to get shape (batch, seq_len, num_heads, d_k)
        #Do this because we want to concatenate the heads together
        head_outputs = head_outputs.transpose(1, 2)

        #Concatenate the heads together using .view()
        #Merges the heads (3rd and 4th dimension) together to get a tensor of shape (batch, seq_len, d_model)
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
        #Retrieve saved tensors and parameters using ctx
        query, key, value, Q, K, V, attn, concat, W_q, W_k, W_v, W_o = ctx.saved_tensors
        num_heads = ctx.num_heads
        scale = ctx.scale
        batch, seq_len, d_model = query.size()
        d_k = d_model // num_heads

        # output = concatenated * W_o  (via einsum: 'btd,dh->bth')
        #So grad_concat =  grad_output * W_o.t() by chain rule
        #grad_output is of shape (batch, seq_len, d_model)
        #W_o is of shape (output_dim, d_model) -> (h, d) -> which in our case is (d_model, d_model)
        #So we get desired shape (batch, seq_len, d_model) for grad_concat which matches our concatenation
        grad_concat = torch.einsum('bth,hd->btd', grad_output, W_o.t())

        # Compute grad_W_o: For the forward operation, output = concatenated * W_o (via 'btd,dh->bth').
        # By the chain rule, grad_W_o = concat.T() * grad_output, summing over batch (b) and token (t) dimensions.
        #Here, concat has shape (batch, seq_len, d_model) and grad_output has shape (batch, seq_len, d_model).
        #The einsum 'btd,bth->dh' implicitly sums over both b and t given our directions, producing a tensor of shape (d_model, d_model) for grad_W_o, so we don't need to transpose concat
        grad_W_o = torch.einsum('btd,bth->dh', concat, grad_output)

        #Similar to the forward pass, we need to reshape grad_concat to get a gradient for each head output
        #We do this by reshaping grad_concat to (batch, seq_len, num_heads, d_k)
        #Then we swap the second and third dimensions to get shape (batch, num_heads, seq_len, d_k) to match our head_outputs
        grad_head_output = grad_concat.view(batch, seq_len, num_heads, d_k)
        grad_head_output = grad_head_output.transpose(1, 2)

        #The einsum string 'bnqk,bnqd->bnkd' instructs PyTorch to:
        #For each batch and head, for each query position (q), sum the attention weights over all keys (indexed by k) * corresponding value vectors
        #In forward pass each value vector contributes to the output at every query position so the gradient for each value is the sum over all query positions of the attention weight times the gradient from the head output
        grad_attn = torch.einsum('bnqd,bnkd->bnqk', grad_head_output, V)
        
        #Here, the attention weights determine how much each value contributes to the head output.
        #This computes the gradient for each value vector V by weighting the gradient from the head outputs by the attention weights.
        #Similar to above
        grad_V = torch.einsum('bnqk,bnqd->bnkd', attn, grad_head_output)

        # Compute the aggregated gradient over keys for each query:
        # For softmax, the derivative is A * (grad_attn - sum(grad_attn * A)).
        # We sum (grad_attn * attn) over the key dimension to get sum_grad.
        sum_grad = torch.sum(grad_attn * attn, dim=-1, keepdim=True)

        # Subtract sum_grad from grad_attn and multiply elementwise by attn to obtain grad_scores,
        # which gives the gradient of the loss with respect to the raw scores.
        grad_scores = attn * (grad_attn - sum_grad)

        # Finally, multiply by the scale factor (used in forward) to complete the gradient computation.
        grad_scores = grad_scores * scale

        # grad_Q: For each query Q[b,n,q,d], the dot product with K[b,n,k,d] affects scores for all keys k.
        # Thus, grad_Q[b,n,q,d] = sum_k (grad_scores[b,n,q,k] * K[b,n,k,d]).
        # The einsum 'bnqk,bnkd->bnqd' multiplies grad_scores and K elementwise over d and sums over k.
        grad_Q = torch.einsum('bnqk,bnkd->bnqd', grad_scores, K)

        # grad_K: For each key K[b,n,k,d], its contribution spans all queries q.
        # Hence, grad_K[b,n,k,d] = sum_q (grad_scores[b,n,q,k] * Q[b,n,q,d]).
        # The einsum 'bnqk,bnqd->bnkd' multiplies grad_scores and Q elementwise over d and sums over q.
        grad_K = torch.einsum('bnqk,bnqd->bnkd', grad_scores, Q)

        #We need to get the gradients for Q, K and V back to the shape (batch, seq_len, d_model)
        #We will do the reverse of what we did in forward pass, swaping the second and third dimensions to get shape (batch, num_heads, seq_len, d_k)
        #Then we concatenate the heads together to get shape (batch, seq_len, d_model)
        grad_Q_reshaped = grad_Q.transpose(1, 2)
        grad_Q_reshaped = grad_Q_reshaped.contiguous().view(batch, seq_len, d_model)

        grad_K_reshaped = grad_K.transpose(1, 2)
        grad_K_reshaped = grad_K_reshaped.contiguous().view(batch, seq_len, d_model)

        grad_V_reshaped = grad_V.transpose(1, 2)
        grad_V_reshaped = grad_V_reshaped.contiguous().view(batch, seq_len, d_model)

        #For each of these linear operations, the gradient with respect to the input is given by 
        #multiplying the gradient of the projection (e.g. grad_Q_reshaped) by the transpose of the corresponding weight matrix.
        #This is by the chain rule
        grad_query = torch.einsum('btd,dh->bth', grad_Q_reshaped, W_q.t())
        grad_key   = torch.einsum('btd,dh->bth', grad_K_reshaped, W_k.t())
        grad_value = torch.einsum('btd,dh->bth', grad_V_reshaped, W_v.t())

        #The gradients with respect to the weight matrices are obtained by the outer product of the original input and the gradient of the output, summed over the batch and token dimensions.
        #Here, the einsum 'btd,bte->de' basically computes query^T * grad_Q_reshaped (and similarly for key and value), because we are summing over the batch and token dimension
        #This results in gradients of shape (d_model, d_model) that match the shapes of W_q, W_k, and W_v.
        grad_W_q = torch.einsum('btd,bte->de', query, grad_Q_reshaped)
        grad_W_k = torch.einsum('btd,bte->de', key,   grad_K_reshaped)
        grad_W_v = torch.einsum('btd,bte->de', value, grad_V_reshaped)

        #Returns a gradient for every input to the forward, for non-tensor inputs (like num_heads), we return None.
        return grad_query, grad_key, grad_value, grad_W_q, grad_W_k, grad_W_v, grad_W_o, None


#Set up testing values

def custom_mha(query, key, value, W_q, W_k, W_v, W_o, num_heads):
    return MultiHeadAttention.apply(query, key, value, W_q, W_k, W_v, W_o, num_heads)
if __name__ == "__main__":
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

    #Compare backwards pass

    inputs = (query, key, value, W_q, W_k, W_v, W_o, num_heads)

    test = torch.autograd.gradcheck(custom_mha, inputs, eps=1e-6, atol=1e-4)
    print("Gradient check passed?", test)