import numpy as np
import matplotlib as plt
import os
#from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
use_cuda = torch.cuda.is_available()
use_cuda
import matplotlib.pyplot as plt
import numpy as np

#Preparar el conjunto de datos
mean = (0.5,)
std = (1.0,)
datapath = './data'
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

train = datasets.MNIST(root=datapath, train=True,  download=True, transform=trans)
test  = datasets.MNIST(root=datapath, train=False, download=True, transform=trans)

print('Número de datos de entrenamiento: {} \nNúmero de datos de prueba: {}'.format(len(train), len(test)))

#Cargador de datos
batch_size = 100
n_iters    = 80000
num_epochs = int( n_iters / (len(train) / batch_size) )

train_loader = torch.utils.data.DataLoader(dataset    = train,
                                           batch_size = batch_size,
                                           shuffle    = True)
test_loader  = torch.utils.data.DataLoader(dataset    = test,
                                           batch_size = batch_size,
                                           shuffle    = False)
                                    
#Crear RNA

def miReLU(x, alpha=1.):
    return x*(x>0)

def miELU(x, alpha=1.):
    y = alpha*(np.exp(x)-1)
    return x*(x>0) + y*(y<0)
   
x = np.array(range(-100,100))/10.
#Instanciamos RELU,ELU,TANH, definiendo el rango del array en x.
yReLU = miReLU(x)
yELU  = miELU(x)
yTanh = np.tanh(x)

#Representacion gráfica RELU, ELU,TANH
plt.figure(figsize=(10,3))
plt.subplot(131)
plt.plot(x, yELU)
plt.axhline(0, linestyle='--', color='k')
plt.title('ELU')

plt.subplot(132)
plt.plot(x, yTanh)
plt.axhline(0, linestyle='--', color='k')
plt.title('Tanh')

plt.subplot(133)
plt.plot(x, yReLU)
plt.axhline(0, linestyle='--', color='k')
plt.title('ReLU')
plt.show()

class MLPnet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPnet, self).__init__()
        '''
        El método init define las capas de las cuales constará el modelo, 
        aunque no la forma en que se interconectan
        '''
        # Función lineal 1: 784 --> 100
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        # Activación no lineal 1: 100 -->100
        self.relu1 = nn.ReLU()
        
        # Función lineal 2: 100 --> 100
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Activación no lineal 1: 100 -->100
        self.tanh2 = nn.Tanh()
        
        # Función lineal 3: 100 --> 100
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # Activación no lineal 3: 100 -->100
        self.elu3 = nn.ELU()
        
        # Función lineal 3: (Capa de salida): 100 --> 10
        self.fc4 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = x.view(-1, input_dim)  # aqui convertimos la imagen a un vector unidimensional
        # Capa 1
        z1 = self.fc1(x)
        y1 = self.relu1(z1)
        # Capa 2
        z2 = self.fc2(y1)
        y2 = self.tanh2(z2)
        # Capa 3
        z3 = self.fc3(y2)
        y4 = self.elu3(z3)
        # Capa 4 (salida)
        out = self.fc4(y4)
        return out
    
    def name(self):
        return "MLP"

#instanciamos la RNA
input_dim  = 28*28  # 784 número de pixeles
hidden_dim = 150    # número de neuronas en las capas ocultas
output_dim = 10     # número de etiquetas

model = MLPnet(input_dim, hidden_dim, output_dim)
#Ya que disponemos de un GPU, pasamos el modelo al espacio de cálculo de la GPU
if use_cuda:
    model = model.cuda()
#Intancia perdida mediante usando antropía cruzada.
error = nn.CrossEntropyLoss()

#Instancia Optimizador, descenso de gradiante
learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#Entrenamiento del modelo
loss_list         = []
iteration_list    = []
accuracy_list     = []
accuracy_list_val = []
f = open("Simulation.txt", "x")

for epoch in range(num_epochs):
    total=0
    correct=0
    # - - - - - - - - - - - - - - - 
    # Entrena la Red en lotes cada época
    # - - - - - - - - - - - - - - - 
    for i, (images, labels) in enumerate(train_loader):
        
        if use_cuda:                              # Define variables
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images) 
        labels = Variable(labels)
        
        optimizer.zero_grad()                      # Borra gradiente
        outputs = model(images)                    # Propagación
        loss    = error(outputs, labels)           # Calcula error
        loss.backward()                            # Retropropaga error
        optimizer.step()                           # Actualiza parámetros
        
        predicted = torch.max(outputs.data, 1)[1]  # etiqueta predicha (WTA)
        total += len(labels)                       # número total de etiquetas en lote
        correct += (predicted == labels).sum()     # número de predicciones correctas
        
    # calcula el desempeño en entrenamiento: Precisión (accuracy)
    accuracy = float(correct) / float(total)
    # almacena la evaluación de desempeño
    iteration_list.append(epoch)
    loss_list.append(loss.item())
    accuracy_list.append(accuracy)

    # - - - - - - - - - - - - - - - 
    # Evalúa la predicción en lotes cada época
    # - - - - - - - - - - - - - - - 
    correct = 0
    total   = 0
    for images, labels in test_loader: 


        if use_cuda:
            images, labels = images.cuda(), labels.cuda()
        images = Variable(images)                   # Define variables
        labels = Variable(labels)
        
        outputs = model(images)                     # inferencia

        predicted = torch.max(outputs.data, 1)[1]   # etiqueta predicha (WTA)  
        total += len(labels)                        # número total de etiquetas en lote
        correct += (predicted == labels).sum()      # número de predicciones correctas

    # calcula el desempeño: Precisión (accuracy)
    accuracy_val = float(correct) / float(total)
    accuracy_list_val.append(accuracy_val)

    # - - - - - - - - - - - - - - - 
    # Despliega evaluación
    # - - - - - - - - - - - - - - - 
    mensaje='Epoch: {:02}  Loss: {:.6f}  Accuracy: {:.6f}  Accuracy Val: {:.6f}'.format(epoch, loss.data, accuracy, accuracy_val)
    print(mensaje)
    f=open('Simulation.txt','a')
    f.write('\n'+mensaje)
    f.close()

# VISUALIZACIÓN
# Loss
plt.plot(iteration_list,loss_list)
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title("Loss Train")
plt.show()
# Accuracy
plt.plot(iteration_list,accuracy_list,'b')
plt.plot(iteration_list,accuracy_list_val, 'g')
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.title("Accuracy: Test - Val ")
plt.show()