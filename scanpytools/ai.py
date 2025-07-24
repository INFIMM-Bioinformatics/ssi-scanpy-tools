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
    duration = round(duration, 2)

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

def prioritize_genes(
    gene_list,
    context,
    api_token,
    n=20,
    platform="huggingface",
    provider="hf-inference",
    llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    max_tokens=2000,
    temperature=0.6
):
    """
    Retrieve and rank the top N genes and their descriptions based on their relevance to a specific biological context
    using various LLM platforms and providers.

    Args:
        gene_list (list of str): A list of gene names to be prioritized and analyzed.
        context (str): The biological context to consider when ranking genes.
        api_token (str): The API token for authentication (used for all platforms).
        n (int, optional): The number of top genes to retrieve. Defaults to 20.
        platform (str): The platform to use ("huggingface", "google", "openai").
        provider (str): The provider for the platform (e.g., "hf-inference" for Hugging Face).
        llm_model (str, optional): The language model to be used for inference.
        max_tokens (int, optional): The maximum number of tokens in the model's response. Defaults to 2000.
        temperature (float, optional): The temperature for the model's response. Defaults to 0.6.

    Returns:
        tuple: A tuple containing:
            - full_reason_process (str): The complete response from the model.
            - top_genes_dict (dict): A dictionary mapping the top N ranked genes to their descriptions.
            - summary_match (list): A concise summary of the cell functional profile.
            - cell_type_match (str): A short description of the cell type.
    """
    # Consolidate the prompt
    gene_list_character = " ".join(gene_list) + " "
    prompt = (
        f"Given the context of {context}, go through all of the genes below and return the top {n} genes "
        "in the format of **gene** - function (Note that the gene should be strictly from the input and in the returned results wrapped between ** and **). "
        "Additionally, provide a concise summary of the cell functional profile enclosed ONLY between <<<SUMMARY_START>>> and <<<SUMMARY_END>>> markers, with no extra text before or after the markers. "
        "Finally, provide 3 to 5 words as if you are to refer the cell type in Nature Immunology, enclosed between <<<CELL_TYPE_START>>> and <<<CELL_TYPE_END>>> markers. "
        f"{gene_list_character}"
    )

    # Initialize variables
    full_reason_process = ""
    top_genes_dict = {}
    summary_match = []
    cell_type_match = ""

    if platform == "huggingface":
        # Hugging Face Inference API
        from huggingface_hub import InferenceClient
        client = InferenceClient(provider=provider, api_key=api_token)
        messages = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        full_reason_process = completion.choices[0].message.content

    elif platform == "google":
        # Google Gemini API
        from google import genai
        client = genai.Client(api_key=api_token)
        response = client.models.generate_content(
            model=llm_model,
            contents=prompt,
        )
        full_reason_process = response.text

    elif platform == "openai":
        # OpenAI GPT-4o API
        from openai import OpenAI
        client = OpenAI()
        response = client.responses.create(
            model=llm_model,
            input=prompt
        )
        full_reason_process = response.output_text

    else:
        raise ValueError(f"Unsupported platform: {platform}")

    # Extract top genes, summary, and cell type
    import re
    try:
        # Extract the top N genes and their descriptions using regex
        pattern = r'\*\*([\w()]+)\*\* - ([^\n]+)'
        top_genes_with_descriptions = re.findall(pattern, full_reason_process)
        top_genes_dict = {gene: description for gene, description in top_genes_with_descriptions}
    except Exception as e:
        print(f"An error occurred while extracting top {n} genes: {e}")

    try:
        # Extract the summary
        summary_pattern = r'<<<SUMMARY_START>>>(.*?)<<<SUMMARY_END>>>'
        summary_match = re.findall(summary_pattern, full_reason_process, re.DOTALL)
        summary_match = summary_match[0].strip() if summary_match else ""
    except Exception as e:
        print(f"An error occurred while extracting summary: {e}")

    try:
        # Extract the cell type
        cell_type_pattern = r'<<<CELL_TYPE_START>>>(.*?)<<<CELL_TYPE_END>>>'
        cell_type_match = re.findall(cell_type_pattern, full_reason_process, re.DOTALL)
        cell_type_match = cell_type_match[0].strip() if cell_type_match else ""
    except Exception as e:
        print(f"An error occurred while extracting cell type: {e}")

    return full_reason_process, top_genes_dict, summary_match, cell_type_match

