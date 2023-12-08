# **Tecnicatura Universitaria en Inteligencia Artificial**
## **Trabajo Práctico Integrador**
### ***Aprendizaje Automatico 1***
---

**Equipo**:
- Ferrucci Constantino
- Giampaoli Fabio

<br>

**Fechas**:
- 13/10/2023 (entrega 1, hasta item 3)
- 27/10/2023 (entrega 2, hasta item 4)

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

## **Resumen**


El dataset elegido consiste de variables meteorologicas de Australia en determinado periodo de tiempo que seran de interes para la predicion y clasificacion de la cantidad y posibilidad de lluvia en algun dia con ciertas caracteristicas mediante modelos estudiados en el cursado de la materia.\
<br>

El dataset cuenta con distintos tipos de varibables (continuas, categoricas, discretas, binarias), muchas de ellas son de utilidad para obtener informacion del climatica del registro (un dia particular), y otras son de utilidad para ubicacion espacio-temporal del registro.\
<br>

En esta primera etapa del trabajo practico, es de principal interes el analisis y tratamiento del conjunto de datos para conocer el contexto sobre el cual vamos a desarrollarnos, y luego realizar predicciones utilizando modelos de regresion lineal con diferentes caracteristicas, y comparar cual tiene mejor comportameinto a la hora de predecir.\
<br>

Es claro que un el modelo por si solo no puede realizar un buen trabajo si el dataset no es preprocesado para maximizar el rendimiento del entrenamiento. Por ello, la primer etapa es el tratamiento primario de los datos, donde se imputan, codifican, suavizan, y estandarizan las variables.\
<br>

Como segunda etapa planteamos el analisis descriptivo de los datos, ya que es de importancia conocer el contexto de los datos e interpretar correctamente como se relacionan las variables entre si.\
<br>

Luego, se realizan las predicciones con los diferentes modelos y se comparan sus rendimientos utilizando metricas de validacion para comprender cual y por que un modelo hace mejor que los demas.\
<br>

Por ultimo, se plantea elegir metodos de optimizcion de hiperparametros, mejorar la explicabilidad del modelo y simular la puesta en produccion de los mismos.


## **Acceso**

Para poder acceder al deploy del proyecto lo puedes visitar en el siguiente link:

Link: https://aprendizajeautomatico1-wzveamkrgdnnyqvyrbrjfp.streamlit.app/

Además puedes visualizar el archivo con sus respectivas salidas directamente desde google colab:

Link: https://drive.google.com/file/d/1Itfa8GS_oLb3NQ77ukn1l7NVXI_yRzZA/view?usp=sharing