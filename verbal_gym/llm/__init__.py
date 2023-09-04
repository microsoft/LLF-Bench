import os

def standardize_model_name(model):
    if model in ('gpt-35', 'gpt-3.5', 'gpt-35-turbo', 'gpt-3.5-turbo'):
        model = 'gpt-35-turbo'
    if model in ('gpt-3'):
        model = 'text-davinci-003'
    return model


def make_llm(model, **kwargs):
    """ model = backend:model_name """

    backend, model = model.split(':')
    available_backends = []
    if os.getenv('AZURE_OPENAI_KEY') is not None:
        available_backends.append('azure')
    if os.getenv('OPENAI_API_KEY') is not None:
        available_backends.append('openai')
    if os.getenv("GCR_GPT_KEY") is not None and os.getenv("GCR_GPT_URL") is not None:
        available_backends.append('gcr')

    if len(available_backends)==0:
        raise Warning("No LLM backend is set up. Please set up at least one backend.")

    if backend not in available_backends:
        # Try to use the first available backend if the requested one is not available
        Warning(f"{backend} backend is not set up. Switch to {available_backends[0]} backend.")
        backend = available_backends[0]

    model = standardize_model_name(model)

    if backend=='gcr':
        if model=='text-davinci-003':
            from verbal_gym.llm.gpt.gpt import GPT3
            return GPT3(model, **kwargs)
        elif model in ('gpt-35', 'gpt-3.5', 'gpt-35-turbo', 'gpt-3.5-turbo'):
            from verbal_gym.llm.gpt.gpt import GPT35
            return GPT35(model, **kwargs)
        else:
            raise ValueError("Unknown LLM model: {}".format(model))

    elif backend in ('azure', 'openai'):
        from verbal_gym.llm.gpt_models import GPT
        return GPT(model=backend+':'+model, **kwargs)

    else:
        raise ValueError("Unknown LLM model: {}".format(model))
