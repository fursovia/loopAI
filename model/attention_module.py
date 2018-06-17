import tensorflow as tf
import numpy as np

def multihead_attention(query_antecedent,
                        memory_antecedent,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        name="multihead_attention",
                        **kwargs):
    """
      Multihead scaled-dot-product attention with input/output transformations.
      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
      Returns:
        The result of the attention transformation. The output shape is
            [batch_size, length_q, hidden_dim]
      Raises:
        ValueError: if the key depth or value depth are not divisible by the
          number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
    with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

        q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                                total_key_depth, total_value_depth,)

        print('k', k.shape)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        q = split_heads(q, num_heads)

        key_depth_per_head = total_key_depth // num_heads

        x = dot_product_attention(
              q,
              k,
              v,)

        x = combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

        x = tf.layers.dense(
            x, output_depth, use_bias=False, name="output_transform")

        return x
    
def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

      # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret

def combine_heads(x):
    """Inverse of split_heads.
      Args:
        x: a Tensor with shape [batch, num_heads, length, channels / num_heads]
      Returns:
        a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))

def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.
      Args:
        x: a Tensor with shape [..., a, b]
      Returns:
        a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])

def dot_product_attention(q,
                          k,
                          v,
                          name=None,):
    """dot-product attention.
      Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
      Returns:
        A Tensor.
    """
    with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        new_q = tf.matmul(weights, v)
        
        return new_q
    
def split_heads(x, num_heads):
        """Split channels (dimension 2) into multiple heads (becomes dimension 1).
      Args:
        x: a Tensor with shape [batch, length, channels]
        num_heads: an integer
      Returns:
        a Tensor with shape [batch, num_heads, length, channels / num_heads]
        """
        return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])
    
def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.
      The first of these two dimensions is n.
      Args:
        x: a Tensor with shape [..., m]
        n: an integer.
      Returns:
        a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = 450 #x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])

def compute_qkv(q,
                m,
                total_key_depth,
                total_value_depth,):
    """Computes query, key and value.
      Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: an integer
      Returns:
        q, k, v : [batch, length, depth] tensors
    """

    q = compute_attention_component(q, total_key_depth,
                                   "q")
    k = compute_attention_component(m, total_key_depth,
                                   "k")
    v = compute_attention_component(m, total_value_depth,
                                   "v")
    return q, k, v

def compute_attention_component(antecedent,
                                total_depth,
                                name="c"):
    """Computes attention compoenent (query, key or value).
    Args:
        antecedent: a Tensor with shape [batch, length, channels]
        total_depth: an integer
        filter_width: An integer specifying how wide you want the attention
          component to be.
        padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        name: a string specifying scope name.
    Returns:
        c : [batch, length, depth] tensor
    """

    return tf.layers.dense(
            antecedent, total_depth, use_bias=False, name=name)