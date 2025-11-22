# CIFAR-10: Un dataset de imágenes para clasificación

Los conjuntos de datos CIFAR-10 y CIFAR-100 son subconjuntos etiquetados del dataset de 80 millones de imágenes pequeñas. CIFAR-10 y CIFAR-100 fueron creados por Alex Krizhevsky, Vinod Nair y Geoffrey Hinton.

## El dataset CIFAR-10

El dataset CIFAR-10 consiste en 60000 imágenes en color de 32x32 píxeles en 10 clases, con 6000 imágenes por clase. Hay 50000 imágenes de entrenamiento y 10000 imágenes de prueba.

El dataset está dividido en cinco lotes de entrenamiento y un lote de prueba, cada uno con 10000 imágenes. El lote de prueba contiene exactamente 1000 imágenes seleccionadas aleatoriamente de cada clase. Los lotes de entrenamiento contienen las imágenes restantes en orden aleatorio, pero algunos lotes de entrenamiento pueden contener más imágenes de una clase que de otra. Entre todos, los lotes de entrenamiento contienen exactamente 5000 imágenes de cada clase.

Las clases en el dataset son:

- **Avión** (airplane)
- **Automóvil** (automobile)
- **Pájaro** (bird)
- **Gato** (cat)
- **Ciervo** (deer)
- **Perro** (dog)
- **Rana** (frog)
- **Caballo** (horse)
- **Barco** (ship)
- **Camión** (truck)

Las clases son completamente mutuamente excluyentes. No hay superposición entre automóviles y camiones. "Automóvil" incluye sedanes, SUVs y vehículos de ese tipo. "Camión" incluye solo camiones grandes. Ninguno de los dos incluye camionetas pickup.

