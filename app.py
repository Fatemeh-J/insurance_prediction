{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "859fdb3b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask,request, url_for, redirect, render_template, jsonify\n",
    "from pycaret.regression import *\n",
    "import pandas as pd\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "import nest_asyncio\n",
    "# import uvicorn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "dc606579",
   "metadata": {},
   "outputs": [],
   "source": [
    "# !pip install uvicorn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "389da545",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Transformation Pipeline and Model Successfully Loaded\n"
     ]
    }
   ],
   "source": [
    "# Assign an instance of the flask class to the variable \"app\".\n",
    "app = Flask(__name__)\n",
    "\n",
    "\n",
    "model = load_model('./Practicing MLOPS/pycaret_deployment')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fb7dfc87",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c0c112dd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0f88e742",
   "metadata": {},
   "outputs": [],
   "source": [
    "# By using @app.get(\"/\") you are allowing the GET method to work for the / endpoint.\n",
    "@app.route(\"/\")\n",
    "def home():\n",
    "#     return \"Congratulations! Your API is working as expected.\"\n",
    "    return render_template('index.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "692abc57",
   "metadata": {},
   "outputs": [],
   "source": [
    "# @app.route('/predict', methods=['POST'])\n",
    "# def predict():\n",
    "#     age = request.form['age']\n",
    "#     sex = request.form['sex']\n",
    "#     income = request.form['income']\n",
    "\n",
    "#     # Call your machine learning model here to get predictions\n",
    "#     # predictions = your_model.predict(age, sex, income)\n",
    "\n",
    "#     return f\"Predicted Value: {predictions}\"\n",
    "\n",
    "\n",
    "@app.route('/predict',methods=['POST'])\n",
    "def predict():\n",
    "    int_features = [x for x in request.form.values()]\n",
    "    final = np.array(int_features)\n",
    "    data_unseen = pd.DataFrame([final], columns = cols)\n",
    "    prediction = predict_model(model, data=data_unseen, round = 0)\n",
    "    prediction = int(prediction.Label[0])\n",
    "    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cb411da1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      "[2023-11-06 14:25:56,366] ERROR in app: Exception on / [GET]\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/app.py\", line 2529, in wsgi_app\n",
      "    response = self.full_dispatch_request()\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/app.py\", line 1825, in full_dispatch_request\n",
      "    rv = self.handle_user_exception(e)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/app.py\", line 1823, in full_dispatch_request\n",
      "    rv = self.dispatch_request()\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/app.py\", line 1799, in dispatch_request\n",
      "    return self.ensure_sync(self.view_functions[rule.endpoint])(**view_args)\n",
      "  File \"/var/folders/81/xmvn25zn01b5s0dgby1klq_40000gn/T/ipykernel_14222/6055064.py\", line 5, in home\n",
      "    return render_template('index.html')\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/templating.py\", line 146, in render_template\n",
      "    template = app.jinja_env.get_or_select_template(template_name_or_list)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/jinja2/environment.py\", line 1081, in get_or_select_template\n",
      "    return self.get_template(template_name_or_list, parent, globals)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/jinja2/environment.py\", line 1010, in get_template\n",
      "    return self._load_template(name, globals)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/jinja2/environment.py\", line 969, in _load_template\n",
      "    template = self.loader.load(self, name, self.make_globals(globals))\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/jinja2/loaders.py\", line 126, in load\n",
      "    source, filename, uptodate = self.get_source(environment, name)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/templating.py\", line 62, in get_source\n",
      "    return self._get_source_fast(environment, template)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/flask/templating.py\", line 98, in _get_source_fast\n",
      "    raise TemplateNotFound(template)\n",
      "jinja2.exceptions.TemplateNotFound: index.html\n",
      "127.0.0.1 - - [06/Nov/2023 14:25:56] \"\u001b[35m\u001b[1mGET / HTTP/1.1\u001b[0m\" 500 -\n",
      "127.0.0.1 - - [06/Nov/2023 14:25:56] \"\u001b[33mGET /favicon.ico HTTP/1.1\u001b[0m\" 404 -\n"
     ]
    }
   ],
   "source": [
    "if __name__ == '__main__':\n",
    "    app.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2b53ea0d",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'uvicorn' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[14], line 8\u001b[0m\n\u001b[1;32m      5\u001b[0m host \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m127.0.0.1\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m      7\u001b[0m \u001b[38;5;66;03m# Spin up the server!    \u001b[39;00m\n\u001b[0;32m----> 8\u001b[0m \u001b[43muvicorn\u001b[49m\u001b[38;5;241m.\u001b[39mrun(app, host\u001b[38;5;241m=\u001b[39mhost, port\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m8000\u001b[39m, root_path\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m/serve\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'uvicorn' is not defined"
     ]
    }
   ],
   "source": [
    "# Allows the server to be run in this interactive environment\n",
    "nest_asyncio.apply()\n",
    "\n",
    "# This is an alias for localhost which means this particular machine\n",
    "host = \"127.0.0.1\"\n",
    "\n",
    "# Spin up the server!    \n",
    "uvicorn.run(app, host=host, port=8000, root_path=\"/serve\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4f62c790",
   "metadata": {},
   "outputs": [],
   "source": [
    "# @app.post()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "4916cdac",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Transformation Pipeline and Model Successfully Loaded\n",
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
      " * Running on http://127.0.0.1:5000\n",
      "\u001b[33mPress CTRL+C to quit\u001b[0m\n",
      " * Restarting with stat\n",
      "Traceback (most recent call last):\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/ipykernel_launcher.py\", line 17, in <module>\n",
      "    app.launch_new_instance()\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/traitlets/config/application.py\", line 991, in launch_instance\n",
      "    app.initialize(argv)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/traitlets/config/application.py\", line 113, in inner\n",
      "    return method(app, *args, **kwargs)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/ipykernel/kernelapp.py\", line 689, in initialize\n",
      "    self.init_sockets()\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/ipykernel/kernelapp.py\", line 328, in init_sockets\n",
      "    self.shell_port = self._bind_socket(self.shell_socket, self.shell_port)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/ipykernel/kernelapp.py\", line 252, in _bind_socket\n",
      "    return self._try_bind_socket(s, port)\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/ipykernel/kernelapp.py\", line 228, in _try_bind_socket\n",
      "    s.bind(\"tcp://%s:%i\" % (self.ip, port))\n",
      "  File \"/Users/fatemeh/anaconda3/envs/python3_8/lib/python3.8/site-packages/zmq/sugar/socket.py\", line 302, in bind\n",
      "    super().bind(addr)\n",
      "  File \"zmq/backend/cython/socket.pyx\", line 564, in zmq.backend.cython.socket.Socket.bind\n",
      "  File \"zmq/backend/cython/checkrc.pxd\", line 28, in zmq.backend.cython.checkrc._check_rc\n",
      "zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:53553')\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[0;31mSystemExit\u001b[0m\u001b[0;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "# # from flask import Flask,request, url_for, redirect, render_template, jsonify\n",
    "# # from pycaret.regression import *\n",
    "# # import pandas as pd\n",
    "# # import pickle\n",
    "# # import numpy as np\n",
    "\n",
    "# # app = Flask(__name__)\n",
    "\n",
    "# model = load_model('./Practicing MLOPS/pycaret_deployment')\n",
    "# cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']\n",
    "\n",
    "# @app.route('/')\n",
    "# def home():\n",
    "#     return render_template(\"home.html\")\n",
    "\n",
    "# @app.route('/predict',methods=['POST'])\n",
    "# def predict():\n",
    "#     int_features = [x for x in request.form.values()]\n",
    "#     final = np.array(int_features)\n",
    "#     data_unseen = pd.DataFrame([final], columns = cols)\n",
    "#     prediction = predict_model(model, data=data_unseen, round = 0)\n",
    "#     prediction = int(prediction.Label[0])\n",
    "#     return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))\n",
    "\n",
    "# @app.route('/predict_api',methods=['POST'])\n",
    "# def predict_api():\n",
    "#     data = request.get_json(force=True)\n",
    "#     data_unseen = pd.DataFrame([data])\n",
    "#     prediction = predict_model(model, data=data_unseen)\n",
    "#     output = prediction.Label[0]\n",
    "#     return jsonify(output)\n",
    "\n",
    "# if __name__ == '__main__':\n",
    "#     app.run(debug=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53423358",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e4e0feef",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "python3_8",
   "language": "python",
   "name": "python3_8"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
