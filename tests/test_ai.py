# Test the function for multiple genes
import scanpytools as sctl
from dotenv import load_dotenv
import os
import unittest

# Load environment variables from .env file
load_dotenv()

class TestPrioritizeGenes(unittest.TestCase):
    """
    Unit tests for the prioritize_genes function.
    """

    def setUp(self):
        """
        Set up common variables for the tests.
        """
        
        # Define a list of genes
        self.gene_list = ['Foxp3', 'Il2ra', 'Ikzf2', 'Ctla4', 'Coro2a', 'Ighm', 'Fam169b', 'Inpp4b',
                          'Pim1', 'Ift80', 'Lclat1', 'Tasp1', 'Ikzf4', 'Dst', 'Tnfrsf4', 'Airn',
                          'Lamc1', 'Smc4', 'Lrrc32', 'Hdac9', 'Tnfrsf9', 'Gm19765', 'Samhd1',
                          'Marchf3', 'Izumo1r', 'Resf1', 'Nt5e', 'Swap70', 'Itga6', 'Mctp1',
                          'Myo1e', 'Cd83', 'Gm4956-1', 'Myo3b', 'Cerk', 'Ptprj', 'Ankrd55', 'Cd27',
                          'Gm28112', 'Fam3c', 'Gpr83', 'Inpp5f', 'Ldlrad4', 'Plcl1', 'Sesn1', 'Wls',
                          'Lrig1', 'Phlpp1', 'Vps54', 'Peak1']
        # Define the context
        self.context = "immune response in T cells"
        self.n = 20
        self.temperature = 0.6
        self.max_tokens = 2000
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.hf_key = os.getenv("HF_TOKEN")

    def test_prioritize_genes_huggingface(self):
        """
        Test prioritize_genes with the Hugging Face platform.
        """
        result = sctl.ai.prioritize_genes(
            gene_list=self.gene_list,
            context=self.context,
            api_token=self.hf_key,
            n=self.n,
            platform="huggingface",
            provider="hf-inference",
            llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[1], dict)  # top_genes_dict
        self.assertIsInstance(result[2], str)  # summary_match
        self.assertIsInstance(result[3], str)  # cell_type_match

    def test_prioritize_genes_google(self):
        """
        Test prioritize_genes with the Google Gemini platform.
        """
        result = sctl.ai.prioritize_genes(
            gene_list=self.gene_list,
            context=self.context,
            api_token=gemini_key,
            n=self.n,
            platform="google",
            provider="google",
            llm_model="gemini-1.5-turbo",
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[1], dict)  # top_genes_dict
        self.assertIsInstance(result[2], str)  # summary_match
        self.assertIsInstance(result[3], str)  # cell_type_match

    def test_prioritize_genes_openai(self):
        """
        Test prioritize_genes with the OpenAI platform.
        """
        result = sctl.ai.prioritize_genes(
            gene_list=self.gene_list,
            context=self.context,
            api_token=openai_key,
            n=self.n,
            platform="openai",
            provider="openai",
            llm_model="gpt-4",
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[1], dict)  # top_genes_dict
        self.assertIsInstance(result[2], str)  # summary_match
        self.assertIsInstance(result[3], str)  # cell_type_match

    def test_prioritize_genes_invalid_platform(self):
        """
        Test prioritize_genes with an invalid platform.
        """
        with self.assertRaises(ValueError):
            sctl.ai.prioritize_genes(
                gene_list=self.gene_list,
                context=self.context,
                api_token=hf_key,
                n=self.n,
                platform="invalid_platform",
                provider="hf-inference",
                llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

    def test_prioritize_genes_empty_gene_list(self):
        """
        Test prioritize_genes with an empty gene list.
        """
        result = sctl.ai.prioritize_genes(
            gene_list=[],
            context=self.context,
            api_token=hf_key,
            n=self.n,
            platform="huggingface",
            provider="hf-inference",
            llm_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result[1]), 0)  # top_genes_dict should be empty
        self.assertEqual(result[2], "")  # summary_match should be empty
        self.assertEqual(result[3], "")  # cell_type_match should be empty


if __name__ == "__main__":
    unittest.main()
