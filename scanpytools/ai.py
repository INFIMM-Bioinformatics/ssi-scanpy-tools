def is_gpu_available():
    """
    Check if a GPU is available in the system (Nvidia or AMD).

    Returns:
        bool: True if a GPU is available, False otherwise.
    """
    import torch
    import subprocess
    # Check for Nvidia GPU
    if torch.cuda.is_available():
        return True
    
    # Check for AMD GPU using ROCm
    try:
        result = subprocess.run(['rocminfo'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    return False

def check_vllm_process():
    """
    Check if the vLLM server process is running and extract the model name if available.

    Returns:
        str: The model name if the vLLM server is running, None otherwise.
    """
    import subprocess
    import re
    # Execute the ps command to list processes
    result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Search for the vllm process
    vllm_process_line = None
    for line in output.splitlines():
        if 'vllm serve' in line:
            vllm_process_line = line
            break

    if vllm_process_line:
        # Extract the model name using regex
        match = re.search(r'vllm serve (\S+)', vllm_process_line)
        if match:
            model_name = match.group(1)
            print(f"vLLM server is running with model: {model_name}")
            return model_name
        else:
            print("vLLM server is running, but model name could not be extracted.")
            return None
    else:
        print("vLLM server is not running.")
        return None

def gene_x_function_in_the_context_of_y_as_Z(gene, context, role="immunologist", run_mode="local", model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", return_format="full", temperature=0.6, include_duration=False):
    """
    Describe the function of a gene in a specific context using an AI model.

    Args:
        gene (str): The gene to describe.
        context (str): The context in which to describe the gene's function.
        role (str, optional): The role to assume in the description. Defaults to "immunologist".
        run_mode (str, optional): The mode to run the function in. Defaults to "local".
        model (str, optional): The AI model to use. Defaults to "deepseek-ai/DeepSeek-R1-Distill-Llama-8B".
        return_format (str, optional): The format of the returned response. Defaults to "full".
        temperature (float, optional): The temperature for the AI model. Defaults to 0.6.
        include_duration (bool, optional): Whether to include the duration of the operation. Defaults to False.

    Returns:
        dict: A dictionary containing the response and optionally the duration.
    """
    from openai import OpenAI
    from datetime import datetime
    if run_mode == "local":
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"

    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    # Generate the prompt
    prompt_question = f"<think>\nUse 20 words to describe gene {gene} function in the context of {context} as a {role}"

    # Record the start time
    start_time = datetime.now()

    # Get the completion
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_question}],
        temperature=temperature
    )

    # Record the end time
    end_time = datetime.now()

    # Calculate the duration
    duration = (end_time - start_time).total_seconds()

    # Prepare the response dictionary
    response_dictionary = response.to_dict()
    assistant_message = response_dictionary['choices'][0]['message']['content']

    # Return the response and optionally the duration
    result = {"response": response_dictionary}
    if return_format == "text":
        result["response"] = assistant_message
    elif return_format == "text_concise":
        result["response"] = assistant_message.split("</think>")[1].strip()

    if include_duration:
        result["duration"] = duration

    return result