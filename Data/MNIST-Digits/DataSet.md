**Descripción**

**Autores:** Yann LeCun, Corinna Cortes, Christopher J.C. Burges
**Fuente:** [Página web de MNIST](http://yann.lecun.com/exdb/mnist/) - Fecha desconocida
**Por favor, cite:**

La base de datos MNIST de dígitos manuscritos con 784 características, datos brutos disponibles en: http://yann.lecun.com/exdb/mnist/. Se puede dividir en un conjunto de entrenamiento de los primeros 60,000 ejemplos y un conjunto de prueba de 10,000 ejemplos.

Es un subconjunto de un conjunto más grande disponible en NIST. Los dígitos han sido normalizados en tamaño y centrados en una imagen de tamaño fijo. Es una buena base de datos para quienes quieren probar técnicas de aprendizaje y métodos de reconocimiento de patrones con datos del mundo real, dedicando un esfuerzo mínimo en el preprocesamiento y formateo.

Las imágenes originales en blanco y negro (binivel) de NIST se normalizaron en tamaño para ajustarse a un cuadro de 20x20 píxeles mientras se preservaba su relación de aspecto. Las imágenes resultantes contienen niveles de gris como resultado de la técnica de anti-aliasing utilizada por el algoritmo de normalización. Las imágenes se centraron en una imagen de 28x28 calculando el centro de masa de los píxeles y trasladando la imagen para posicionar este punto en el centro del campo de 28x28.

Con algunos métodos de clasificación (particularmente métodos basados en plantillas, como SVM y K-vecinos más cercanos), la tasa de error mejora cuando los dígitos se centran mediante un cuadro delimitador en lugar del centro de masa. Si realiza este tipo de preprocesamiento, debe informarlo en sus publicaciones.

La base de datos MNIST se construyó a partir de los conjuntos de NIST. Originalmente, NIST designó SD-3 como su conjunto de entrenamiento y SD-1 como su conjunto de prueba. Sin embargo, SD-3 es mucho más limpio y fácil de reconocer que SD-1. La razón de esto se encuentra en el hecho de que SD-3 se recopiló entre empleados de la Oficina del Censo, mientras que SD-1 se recopiló entre estudiantes de secundaria. Sacar conclusiones sensatas de los experimentos de aprendizaje requiere que el resultado sea independiente de la elección del conjunto de entrenamiento y prueba entre el conjunto completo de muestras. Por lo tanto, fue necesario construir una nueva base de datos mezclando los conjuntos de datos de NIST.

El conjunto de entrenamiento de MNIST se compone de 30,000 patrones de SD-3 y 30,000 patrones de SD-1. Nuestro conjunto de prueba se compuso de 5,000 patrones de SD-3 y 5,000 patrones de SD-1. El conjunto de entrenamiento de 60,000 patrones contenía ejemplos de aproximadamente 250 escritores. Nos aseguramos de que los grupos de escritores del conjunto de entrenamiento y del conjunto de prueba fueran disjuntos.

SD-1 contiene 58,527 imágenes de dígitos escritos por 500 escritores diferentes. En contraste con SD-3, donde los bloques de datos de cada escritor aparecían en secuencia, los datos en SD-1 están mezclados. Las identidades de los escritores para SD-1 están disponibles y usamos esta información para ordenar los datos por escritor. Luego dividimos SD-1 en dos: los caracteres escritos por los primeros 250 escritores fueron a nuestro nuevo conjunto de entrenamiento. Los 250 escritores restantes se colocaron en nuestro conjunto de prueba. Así, tuvimos dos conjuntos con aproximadamente 30,000 ejemplos cada uno.

El nuevo conjunto de entrenamiento se completó con suficientes ejemplos de SD-3, comenzando en el patrón #0, para formar un conjunto completo de 60,000 patrones de entrenamiento. De manera similar, el nuevo conjunto de prueba se completó con ejemplos de SD-3 comenzando en el patrón #35,000 para formar un conjunto completo de 60,000 patrones de prueba.

Solo un subconjunto de 10,000 imágenes de prueba (5,000 de SD-1 y 5,000 de SD-3) está disponible en este sitio. El conjunto completo de entrenamiento de 60,000 muestras está disponible.
