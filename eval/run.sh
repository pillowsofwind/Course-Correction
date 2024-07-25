mkdir output
mkdir raw_results
python eval_data.py --model llama2 ## specify your model here, valid option = ["llama2","vicuna","zephyr","llama3","glm","qwen_05","qwen_15","qwen_7","qwen_72","phi"]
python gpteval.py --model llama2 --openai_key your_api_key_here #please specify the model, valid option = ["llama2","vicuna","zephyr","llama3","glm","qwen_05","qwen_15","qwen_7","qwen_72","phi"], and your openai API key