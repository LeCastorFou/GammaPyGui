from Main import create_app
import socket
from Main.BackEnd.controller import  pwa

app = create_app()

app.register_blueprint(pwa.bp)
# Pour ne pas passer par les variables d'environement

app.jinja_env.cache = {}


if socket.gethostname() in []:
    if __name__ == '__main__':
        app.run(debug=True,host= '0.0.0.0', port = 80)
else:
    if __name__ == '__main__':
        app.run(debug=True, port = 5040)
