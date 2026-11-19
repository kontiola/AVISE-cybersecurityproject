Configuring Connectors
=================================

Connectors are a component in the AVISE framework that contain the logic
for making requests and receiving responses from AI models. When using AVISE,
you need to pass a Connector configuration JSON file via the CLI argument ``--connectorconf``.

AVISE comes with multiple prebuilt configuration files for commonly used AI model
hosting services:

Ollama Connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Ollama <https://ollama.com/>`__ is a widely used software for running language models.
To use AVISE with Ollama models, you can modify the existing ``avise/configs/connector/languagemodel/ollama.json``
configuration file:

.. code-block:: json
   :caption: ``avise/configs/connector/languagemodel/ollama.json``

   {
       "target_model": {
           "connector": "ollama-lm",
           "type": "language_model",
           "name": "phi3:latest", //ADD NAME OF THE OLLAMA MODEL TO TEST HERE
           "api_url": "http://localhost:11434", //Ollama default
           "api_key": null
       }
   }

In the Connector configuraiton files, ``target_model`` defines the configurations for the
model you wish to test and evaluate. ``target_model`` requires the following subfields:

* ``"connector"``: Name of the connector to use (See available connectors by running CLI command ``avise --connector_list``)
* ``"type"``: Type of the AI model *(e.g. language_model, continual_learning, multi_modal)*
* ``"name"``: Name of the model *(For Ollama connector, use the ollama model name)*
* ``"api_url"``: URL for the API endpoint (``http://localhost:11434`` is the default when running Ollama models locally)
* ``"api_key"``: Optional API authorization key if required by the API Endpoint. **Leave as ``null`` if not needed**



OpenAI Connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use AVISE with `OpenAI <https://openai.com/>`__ models, you can modify the existing ``avise/configs/connector/languagemodel/openai.json``
configuration file:

.. code-block:: json
   :caption: ``avise/configs/connector/languagemodel/openai.json``

    {
        "target_model": {
            "connector": "openai-lm",
            "type": "language_model",
            "name": "gpt-4o-mini",
            "api_url": null,
            "headers": null,
            "api_key": "YOUR_OPENAI_API_KEY_HERE"
        }
    }

In the Connector configuration files, ``target_model`` defines the configurations for the
model you wish to test and evaluate. ``target_model`` requires the following subfields:

* ``"connector"``: Name of the connector to use (See available connectors by running CLI command ``avise --connector_list``)
* ``"type"``: Type of the AI model *(e.g. language_model, continual_learning, multi_modal)*
* ``"name"``: Name of the model *(For OpenAI connector, use the openai model name)*
* ``"api_url"``: URL for the API endpoint
* ``"headers"``: Optional Headers for the API request.
* ``"api_key"``: OpenAI Authentication API Key.





Gemini Connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use AVISE with Google `Gemini <https://ai.google.dev/>`__ models, you can modify the existing ``avise/configs/connector/languagemodel/gemini.json``
configuration file:

.. code-block:: json
   :caption: ``avise/configs/connector/languagemodel/gemini.json``

    {
        "target_model": {
            "connector": "gemini-lm",
            "type": "language_model",
            "name": "gemini-2.5-flash",
            "api_url": null,
            "headers": null,
            "api_key": "YOUR_GEMINI_API_KEY_HERE",
            "max_tokens": 5000,
            "completion_kwargs": {}
        }
    }

You can create an API key in `Google AI Studio <https://aistudio.google.com/apikey>`__. Instead of writing the key
into the configuration file, you can leave ``"api_key"`` empty (``""``) or ``null`` and set it as an environment variable:

.. code-block:: bash

   export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"

You can also use the predefined configuration path ``gemini_lm`` with the ``--connectorconf`` argument.

In the Connector configuration files, ``target_model`` defines the configurations for the
model you wish to test and evaluate. ``target_model`` accepts the following subfields:

* ``"connector"``: Name of the connector to use (See available connectors by running CLI command ``avise --connector_list``)
* ``"type"``: Type of the AI model *(e.g. language_model, continual_learning, multi_modal)*
* ``"name"``: Name of the model. **Required.** *(For Gemini connector, use the Gemini model name, e.g.* ``gemini-2.5-flash`` *)*
* ``"api_url"``: Optional URL for the API endpoint. **Leave as** ``null`` **to use the default Google GenAI endpoint.**
* ``"headers"``: Optional headers for the API request.
* ``"api_key"``: Gemini API key. If empty or ``null``, the key is read from the ``GEMINI_API_KEY`` environment variable.
* ``"max_tokens"``: Optional maximum number of tokens to generate. Defaults to ``512``. Newer Gemini models count "thinking" tokens toward this limit, so a higher value (e.g. ``5000``) is recommended to avoid cut-off or empty responses.
* ``"completion_kwargs"``: Optional extra generation settings passed to the Gemini API (e.g. ``{"top_p": 0.9}``). Defaults to ``{}``.



Generic REST API Connector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the Generic REST API Connector, you can connect AVISE to any RESTful API Endpoint. To configure the Generic REST API Connector,
modify the existing ``avise/configs/connector/languagemodel/genericrest.json`` configuration file:

.. code-block:: json
   :caption: ``avise/configs/connector/languagemodel/genericrest.json``

    {
        "target_model": {
            "connector": "generic-rest-lm",
            "type": "language_model",
            "name": "<PLACEHOLDER_NAME>",
            "api_url": "PLACEHOLDER_URL",
            "headers": null,
            "api_key": null,
            "response_field": "KEY_OF_GENERATED_RESPONSE",
            "method": "POST"
        }
    }

In the Connector configuraiton files, ``target_model`` defines the configurations for the
model you wish to test and evaluate. ``target_model`` accepts the following subfields:

* ``"connector"``: Name of the connector to use (See available connectors by running CLI command ``avise --connector_list``)
* ``"type"``: Type of the AI model *(e.g. language_model, continual_learning, multi_modal)*
* ``"name"``: Name of the model. *(Optional for Generic REST Connector)*
* ``"api_url"``: URL for the API endpoint
* ``"headers"``: Optional Headers for the API request.
* ``"api_key"``: OpenAI Authentication API Key.
* | ``"response_field"``:  The field in which the response of the AI model is contained in the API request response. E.g. if the API response is:
  | ``{status: "success", message: "example message", data: {"model_name": "Gemini 2.5", "prompt": "Hi, what is your name?", "response": "Hello there! I'm Gemini, developed by Google."}}``, then the ``response_field`` would be ``["data"]["response"]``.
* ``"method"``: API Method to use.
