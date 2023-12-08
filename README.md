# **Tecnicatura Universitaria en Inteligencia Artificial**
## **Trabajo Práctico Integrador**
### ***Aprendizaje Automatico 1***
---

**Equipo**:
- Ferrucci Constantino
- Giampaoli Fabio

<br>

**Fechas**:

- 13/10/2023 (Primer entrega. Hasta item 3)
- 27/10/2023 (Segunda entrega. Hasta item 4)
- 20/11/2023 (Tercer entrega. Hasta ítem 5 y 6)
- 7/12/2023 (Cuarta entrega. Hasta items 7, 8, 9 y 10)


---

## **Instalación**

Para ejecutar estos programas, se debe clonar el repositorio dentro de su local y posteriormente instalar las librerías necesarias para su ejecución. 

Para ello, se recomienda crear un entorno virtual con `python` de la siguiente forma:

```bash
python -m venv venv
```

Luego activar el entorno virtual con:

```bash
source venv/bin/activate
```
_(En caso de utilizar Windows, el comando anterior cambia a `venv\Scripts\activate.bat`)_

Luego instalar las librerías mediante `pip`:

```bash
pip install -r requirements.txt
```

Una vez instaladas las librerías, se puede ejecutar el programa mediante:

```bash
streamlit run main.py
```

---


## **Acceso**

Para poder acceder al deploy de la aplicación lo puedes visitar en el siguiente link:

*Link*: https://aprendizajeautomatico1-wzveamkrgdnnyqvyrbrjfp.streamlit.app/


Para visualizar el desarrollo del proyecto descrito en el resumen debajo, puede acceder al entorno de google colab:

*Link*: https://drive.google.com/file/d/1Itfa8GS_oLb3NQ77ukn1l7NVXI_yRzZA/view?usp=sharing

---


## **Resumen**

El dataset elegido consiste de variables meteorológicas de Australia en determinado período de tiempo que serán de interes para la predición y clasificación de la cantidad y posibilidad de lluvia en algún día con ciertas características mediante modelos estudiados en el cursado de la materia.\
<br>

El dataset cuenta con distintos tipos de varibables (continuas, categoricas, discretas, binarias, fechas), muchas de ellas son de utilidad para obtener información climática del registro (un día particular), y otras son de utilidad para úbicacion espacio-temporal del registro.\
<br>

En la primera etapa del trabajo práctico, es de principal interés el análisis y tratamiento del conjunto de datos para conocer el contexto sobre el cual vamos a desarrollarnos. Se establecen lineamientos generales y limitaciones para contextualizar a los modelos solo en deteerminadas area de Australia para cierto periodo de tiempo.\
<br>

Es claro que un el modelo por si solo no puede realizar un buen trabajo si el dataset no es preprocesado para maximizar el rendimiento del entrenamiento. Por ello, la segunda etapa es el tratamiento primario de los datos, donde se imputan, codifican, suavizan, estandarizan, se balancean las clases, se crean nuevas variables y se adapta el dataset original para optimizar el eprendizaje de los modelos.\
<br>

La tercer etapa consiste en el entrenamiento y optmización de modelos de regresión lineal para la estimación de la variable continua de la cantidad de lluvia del dia posterior, utilizando gráficas, análisis y descripciones se encuentra un modelo óptimo para este contexto con la limitación de utilizar modelos lineales.\
<br>

La cuarta etapa consiste en el entrenamiento y optmización de modelos de regresión logistica para la estimación de la variable binaria de la presencia de lluvia del dia posterior. Nuevamente, mediante análisis e hiperparametrización se encuentra un modelo de clasificación óptimo, que resulta ser nuestro modelo final para el problema de clasificación.\
<br>

La anteultima etapa trata de resolver los dos problemas previos pero con un enfoque de modelos conexionistas de aprendizaje profundo. Para ello se han diseñado, optimizado y entrenado modelos de redes neuronales, primeramene para la estimación de la variable continua, y luego con un  modelo similar, para la estimación de la variale binaria. Se han encontrado resultados positivos que mejoran los rendimientos para el problema de regresión. Por lo que este modelo es el elegido para la aplicación final. Pero no es el caso del modelo de clasifación, ya que ha demostrado mejor rendimiento el modelo de regresión logística para este problema.\
<br>


Por último, se pretende desplegar una aplicación dentro de un servidor cuyo objetivo es que un usuario pueda ingresar datos metereológicos de ciudades de Australia en un formulario con datos limitados al contexto inicial, y que se pueda elejir el problema a resolver por la aplicación (clasificación o regresión) en base a los datos. Los datos ingresados se procesan y se realizan con ellos predicciones con los modelos entrenados para retornar en tiempo real una predicción.\
<br>

---

