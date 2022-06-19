import torch
import torch.nn as nn
from embedding import TransformerEmbedding

class ScaledDotProductAttention(nn.Module):
	def __init__(self, mask = None):
		super(ScaledDotProductAttention, self).__init__()
		self.mask = mask

	def forward(self, value, key, query):
		batch_size, head, length, tensorDimension = key.size()
		x = torch.dot(query, key.transpose()) / math.sqrt(tensorDimension)

		if self.mask is not None:
			x = x.masked_fill(self.mask == 0, float("-1e20"))

		x = torch.softmax(x, dim=3)
		x = torch.dot(x, value)

		return x

class MultiHeadAttention(nn.Module):
	def __init__(self, embedSize, heads, mask = None):
		super(MultiHeadAttention, self).__init__()
		self.embedSize = embedSize
		self.heads = heads
		self.headDim = embedSize // heads

		assert(headDim * heads == embedSize), "Embed size must divisible by head"

		self.value = nn.Linear(self.headDim, self.headDim, bias = False)
		self.key = nn.Linear(self.headDim, self.headDim, bias = False)
		self.query = nn.Linear(self.headDim, self.headDim, bias = False)
		self.fc_out = nn.Linear(self.heads * self.headDim, self.embedSize)
		self.attention = ScaledDotProductAttention(self.embedSize, mask)

	def split(self, tensor):
		batch_size, length, modelDimension = tensor.size()

		tensorDimension = modelDimension // self.heads
		tensor = tensor.view(batch_size, length, self.heads, tensorDimension).transpose(1, 2)

		return tensor

	def concat(self, tensor):
		batch_size, head, length, tensorDimension = tensor.size()

		modelDimension = head * tensorDimension
		tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, modelDimension)
		return tensor

	def forward(self, value, key, query):
		value = self.split(value)
		key = self.split(key)
		query = self.split(query)

		Vx = self.value.forward(value)
		Kx = self.key.forward(key)
		Qx = self.query.forward(query)

		x = self.attention.forward(Vx, Kx, Qx)
		x = self.concat(x)
		x = self.fc_out.forward(x)
		return x

class LayerNorm(nn.Module):
	def __init__(self, nodes):
		super(LayerNorm).__init__()
		self.layerNorm = nn.LayerNorm(nodes)

	def forward(self, skipLayer, attentionOutput):
		x = self.layerNorm.forward((skipLayer + attentionOutput))

		return x

class PositionWiseFeedForwardNetworks(nn.Module):
	def __init__(self, inputDim, hidden):
		super(PositionWiseFeedForwardNetworks).__init__()
		self.linear1 = nn.Linear(inputDim, hidden)
		self.relu = nn.relu()
		self.linear2 = nn.Linear(hidden, inputDim)

	def forward(self, x):
		x = self.linear1.forward(x)
		x = self.relu.forward(x)
		x = self.linear2.forward(x)

		return x

class EncoderBlock(nn.Module):
	def __init__(self, embedSize, heads, hidden):
		super(EncoderBlock, self).__init__()
		self.multiHeadAttention = MultiHeadAttention(embedSize, heads)
		self.layerNorm1 = LayerNorm()
		self.feedForward = PositionWiseFeedForwardNetworks(embedSize, hidden)
		self.layerNorm2 = LayerNorm()

	def forward(self, x):
		_x = x
		x = self.multiHeadAttention(x,x,x)
		x = self.layerNorm1(_x, x)

		_x = x
		x = self.feedForward(x)
		x = self.layerNorm2(_x, x)
		return x

class DecoderBlock(nn.Module):
	def __init__(self, embedSize, heads, hidden, mask):
		super(DecoderBlock, self).__init__()
		self.maskedMultiHeadAttention = MultiHeadAttention(embedSize, heads, mask = mask)
		self.layerNorm1 = LayerNorm()
		self.multiHeadAttention = MultiHeadAttention(embedSize, heads)
		self.layerNorm2 = LayerNorm()
		self.feedForward = PositionWiseFeedForwardNetworks(embedSize, hidden)
		self.layerNorm3 = LayerNorm()

	def forward(iself, x, x2):
		_x = x
		x = self.maskedMultiHeadAttention(x,x,x)
		x = self.layerNorm1(_x, x)

		_x = x
		x = self.multiHeadAttention(x2,x2,x)
		x = self.layerNorm2(_x, x)

		_x = x
		x = self.feedForward(x)
		x = self.layerNorm3(_x, x)
		return x

class Encoder(nn.Module):
	def __init__(self, enc_voc_size, max_len, embedSize, heads, hidden, n_layers, drop_prob, device):
		super(Encoder, self).__init__()
		self.emb = TransformerEmbedding(d_model=embedSize, max_len=max_len, vocab_size=enc_voc_size, drop_prob=drop_prob, device=device)
		self.layers = nn.ModuleList([EncoderBlock(embedSize=embedSize,hidden=hidden,heads=heads) for _ in range(n_layers)])

	def forward(self, x, src_mask):
		x = self.emb(x)

		for layer in self.layers:
			x = layer(x, src_mask)

		return x

class Decoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, embedSize, heads, hidden, mask, n_layers, drop_prob, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(d_model = embedSize, drop_prob = drop_prob, max_len = max_len, vocab_size = dec_voc_size, device = device)
        self.layers = nn.ModuleList([DecoderBlock(embedSize = embedSize, hidden = hidden, heads = heads) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        output = self.linear(trg)

        return output

class Trasformer(nn.Module):
	def __init__(self, enc_voc_size, max_len, embedSize, heads, hidden, mask, n_layers, drop_prob, device):
        super(Trasformer, self).__init__()

        self.decoder = Decoder(enc_voc_size, max_len, embedSize, heads, hidden, mask, n_layers, drop_prob, device)
        self.enncoder = Encoder(enc_voc_size, max_len, embedSize, heads, hidden, n_layers, drop_prob, device)

     def forwward(self, x):
     	x = self.encoder(x)
     	x = self.decoder(x)

     	return x