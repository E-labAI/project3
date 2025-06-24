Instrucciones:
1. Validar que tengas instalado la versión de python 3.10.11

2. Crear una maquina virtual
python -m venv venv

3. Activar la maquina
venv/Scripts/Activate

4. colocar las siguientes lineas que instalan las librerias necesarias para correr el programa
   - pip install tensorflow
   - pip install matplotlib
   - pip install gradio

5. Descargar las carpetas train, test y valid del siguiente repositorio:
https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species

6. abrir un poweshell terminal en visual studio y ejecutar lo siguiente:
venv/Scripts/Activate
python app.py

cuando tengan todo listo tienen que correr el archivo "app.py", porque si le dan en abrir el server live al index no les va a funcionar, entonces cuando corrar el archivo "app.py"
y todo les funciona bien debe aparecer el siguiente mensaje:


 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 279-022-806


ahi tienen que oprimir la tecla control y darle click donde dice "http://127.0.0.1:5000", con eso les va a abrir la pagina del index y asi si pueden llenar los datos necesarios para que la IA les de la informacion que quieran