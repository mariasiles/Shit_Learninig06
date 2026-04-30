"""CNN encoder + LSTM decoder for image captioning (yunjey-style, modernized)."""
from __future__ import annotations # anotacions modernes

import torch # per crear tensors
import torch.nn as nn # xarxes neuronals de torch
import torchvision.models as models # models preentrenats de torchvision
from torch.nn.utils.rnn import pack_padded_sequence # per teballar amb seqüències de text de longitud variable

# l'EncoderCNN utilitza un ResNet preentrenat per agafar imatges (input) i extreure un vector de característiques (output) de mida `embed_size`. 

class EncoderCNN(nn.Module): # com q hereta pto tenir: model.parameters(), .to(device), .train(), .eval()...
    """ResNet-50 encoder. Outputs an embedding of size `embed_size`."""

    def __init__(self, embed_size: int = 256, backbone: str = "resnet50"): # embed_size és la mida del vector de característiques de la imatge
                                                                           # backbone és el tipus de CNN preentrenat a utilitzar (resnet50 o resnet152)
        super().__init__() # obligatori per inicialitzar la classe base nn.Module. Es creen capes internes self.cnn, self.linear, self.bn
        if backbone == "resnet50": 
            weights = models.ResNet50_Weights.IMAGENET1K_V2 # carrega la configuració de pesos (la xarxa ja sabrà detectar vores, textures, formes, parts d'objectes...)
            net = models.resnet50(weights=weights) # crea la xarxa amb els pesos
        elif backbone == "resnet152":
            weights = models.ResNet152_Weights.IMAGENET1K_V2
            net = models.resnet152(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        modules = list(net.children())[:-1]  # net.children() retorna les capes de la xarxa com a iterador; list() les converteix a llista; [:-1] elimina l'última capa (la de classificació)
        self.cnn = nn.Sequential(*modules) # crea una xarxa seqüencial amb les capes restants --> self.cnn = conv1. + bn1 + relu + maxpool + layer1 + ... + avgpool
        self.linear = nn.Linear(net.fc.in_features, embed_size) # net.fc.in_features és la mida de les característiques d'entrada a la capa de classificació original (2048 per resnet50); embed_size és la mida de les característiques de sortida que volem (256)
                                            # aquesta capa lineal converteix de [B, 2048] a [B, embed_size]
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01) # normalització de batch per estabilitzar l'entrenament i millorar la convergència. momentum=0.01 significa que es farà una mitjana mòbil amb un factor de 0.01 per actualitzar les estadístiques de la normalització

        # Freeze CNN backbone (only fine-tune linear + bn)
        for p in self.cnn.parameters():
            p.requires_grad = False # no entrenarem els pesos de la CNN, només els de la capa lineal i la de batchnorm (simplement no calculem els gradients per aquells pesos)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(): # tot el que està dins d'aquest bloc no calcula gradients (reforça que CNN està congelada)
            features = self.cnn(images)            # entra [B, 3, 224, 224] i surt [B, C, 1, 1]
        features = features.flatten(1)             # [B, C]              # aplana el tensor a partir de la dimensió 1
        features = self.bn(self.linear(features))  # [B, embed_size]     # self.linear converteix de [B, C] a [B, embed_size]; self.bn normalitza vector
        return features # retorna aquest vector final [B, embed_size] que representa les característiques de la imatge


# DecoderRNN és de tipus LSTM i genera captions (output) a partir del vector de característiques (input).
class DecoderRNN(nn.Module):
    """LSTM decoder conditioned on image features (fed once as the first input)."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int,
                 num_layers: int = 1, max_seq_length: int = 20, dropout: float = 0.5):
        # en ordre: mida vector característiques (i dels embeddings de les paraules), mida de l'estat ocult, num de tokens del vocabulari, nombre de capes de l'LSTM apilades (profunditat), longitud màxima de les captions generades
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # crea una capa d'embedding que converteix els ids de les paraules (de 0 a vocab_size-1) en vectors d'embedding de mida embed_size: [B, T] -> [B, T, embed_size]
                                                          # internament conté una matriu (p.ex. [2980, 256]) on cada fila és el vector d'una paraula.
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # crea una capa LSTM que processa seqüències d'embeddings. input_size=embed_size (mida dels vectors d'embedding), hidden_size=mida de l'estat ocult, num_layers=número de capes apilades, batch_first=True significa que les dimensions d'entrada i sortida seran [B, T, ...] en lloc de [T, B, ...]
        self.linear = nn.Linear(hidden_size, vocab_size) # capa final que transforma cada estat ocult de la LSTM en una predicció de token de mida vocab_size.
        self.max_seq_length = max_seq_length # guarda la len màxima de les frases

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: list[int]):
        # features: [B, embed_size] (vector de característiques de la imatge)
        # captions: [B, T] (seqüència de tokens de les captions) [[1,34,56,2,0],[1,3,2,0,0]]
        # lens de cada caption 
        embeddings = self.dropout(self.embed(captions)) # converteix els ids de les paraules en vectors. Abans captions: [B, T]; després embeddings: [B, T, embed_size]; després aplica dropout per regularització
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # features té forma [B, embed_size]; features.unsqueeze(1) -> [B, 1, embed_size]; després concatena al principi de la seqüència d'embeddings: [B, 1 + T, embed_size]
        # captions already include <start>; lengths are full caption lengths
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) # empaqueta les seqüències perquè LSTM ignori el padding mapejat amb la llista lengths, si no LSTM processaria el padding com a part d seqüència
        hiddens, _ = self.lstm(packed) # passa les seqüències empaquetades per la LSTM. hiddens és un tensor empaqeutat amb els estats ocults de la LSTM per cada pas. No fem servir els estats finals.
        outputs = self.linear(self.dropout(hiddens.data)) # hiddens.data desempaqueta tensor, sense padding [sum(lengths), hidden_size]; li aplica dropout; i passa per self.linear que converteix cada estat ocult en una predicció de token: [sum(lengths), vocab_size]
        return outputs # retorna prediccions del decoder. Durant l'entrenament aquestes sortides es comparen amb el target.

    @torch.no_grad() # aquest mètode no calcularà gradients
    def sample(self, features: torch.Tensor, states=None) -> torch.Tensor: 
        # features: [B, embed_size] (vector de característiques de la imatge)
        # states serien els estats interns inicials de la LSTM. A zero per defecte.
        """Greedy generation. Returns [B, max_seq_length] token ids."""
        # greedy generation vol dir que a cada pas escull la paraula amb probabilitat més alta. No fa alternatives com beam_search.
        sampled = [] # anirem guardant els tensors dels tokens generats
        inputs = features.unsqueeze(1) # prepara el primer input per la LSTM. [B, embed_size] -> [B, 1, embed_size], perquè LSTM espera una seqüència (aquí de longitud 1) d'embeddings com a input
        for _ in range(self.max_seq_length): # bucle de generació. Es repetirà max_seq_length vegades
            hiddens, states = self.lstm(inputs, states) # passa l'input actual per la LSTM. La 1a iteració input és el vector de la imatge, les següents l'imput és el vector de la paraula anterior. hidden i state es van actualitzant a cada pas.
            outputs = self.linear(hiddens.squeeze(1)) # [B, 1, hidden_size] -> [B, hidden_size]; després passa per self.linear que converteix a [B, vocab_size], on cada element és la predicció de token per a cada mostra del batch
            _, predicted = outputs.max(1) # tria la paraula amb predicció més alta.
            sampled.append(predicted) 
            inputs = self.embed(predicted).unsqueeze(1) # converteix els ids de les paraules a vectors d'embedding per al següent pas. predicted: [B] -> self.embed(predicted): [B, embed_size] -> unsqueeze(1) -> [B, 1, embed_size]   
        return torch.stack(sampled, dim=1) # converteix la llista de tokens a un tensor [B, max_seq_length] cada fila és una caption.
    

#ENTRENAMENT 
#    images
#[B, 3, 224, 224]
#        |
#        v
#  EncoderCNN
#        |
#        v
#     features
#  [B, embed_size]
#        |
#        v
#   DecoderRNN.forward(features, captions, lengths)
#        |
#        v
#     outputs
# [sum(lengths), vocab_size]

#INFERÈNCIA 
#    images
#[B, 3, 224, 224]
#        |
#        v
#  EncoderCNN
#        |
#        v
#     features
#  [B, embed_size]
#        |
#        v
#   DecoderRNN.sample(features)
#        |
#        v
#     sample_ids
# [B, max_seq_length]



# batch_size = 32
#embed_size = 256
#hidden_size = 512
#vocab_size = 2982
#max_seq_length = 20

# EncoderCNN
# imatges: [32, 3, 224, 224]
# després de ResNet: [32, 2048, 1, 1]
# després de flatten: [32, 2048]
# després de linear + bn: [32, 256]

# DecoderRNN
# captions: [32, 20]
# després de embed: [32, 20, 256]
# després de concatenar features: [32, 21, 256] (+1 per la id de la imatge)
# després de LSTM: [sum(lengths), 512] (desempaquetat, sense padding)
# després de linear: [sum(lengths), 2982]