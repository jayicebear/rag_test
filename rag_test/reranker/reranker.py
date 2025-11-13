

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenReranker:
    """Qwen3-Reranker-4B wrapper for document reranking"""
    
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        max_length: int = 8192,
        device: str = None,
        use_flash_attention: bool = False
    ):
        """
        Initialize Qwen Reranker
        
        Args:
            model_name: HuggingFace model name
            max_length: Maximum sequence length
            device: Device to load model on (None for auto-detection)
            use_flash_attention: Whether to use flash attention 2
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self._load_model(use_flash_attention)
        self._setup_tokens()
        
    def _load_model(self, use_flash_attention: bool):
        """Load tokenizer and model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side='left'
        )
        
        if use_flash_attention:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                attn_implementation="flash_attention_2"
            ).to(self.device).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            ).to(self.device).eval()
    
    def _setup_tokens(self):
        """Setup special tokens and prefixes"""
        # Token IDs for yes/no
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        
        # System prompt template
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query "
            "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        
        # Encode prefix/suffix tokens
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
    
    def format_instruction(
        self, 
        query: str, 
        document: str, 
        instruction: str = None
    ) -> str:
        """
        Format query and document into instruction format
        
        Args:
            query: Search query
            document: Document to rank
            instruction: Task instruction (optional)
            
        Returns:
            Formatted instruction string
        """
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )
    
    def _process_inputs(self, pairs: list) -> dict:
        """
        Tokenize and prepare inputs for model
        
        Args:
            pairs: List of formatted instruction strings
            
        Returns:
            Tokenized inputs ready for model
        """
        # Tokenize
        inputs = self.tokenizer(
            pairs, 
            padding=False, 
            truncation='longest_first',
            return_attention_mask=False, 
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        
        # Add prefix and suffix tokens
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        
        # Pad and move to device
        inputs = self.tokenizer.pad(
            inputs, 
            padding=True, 
            return_tensors="pt", 
            max_length=self.max_length
        )
        
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        return inputs
    
    def _compute_scores(self, inputs: dict) -> list:
        """
        Compute relevance scores from model logits
        
        Args:
            inputs: Tokenized inputs
            
        Returns:
            List of relevance scores (0-1)
        """
        with torch.no_grad():
            batch_scores = self.model(**inputs).logits[:, -1, :]
            
            # Extract yes/no logits
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            
            # Stack and apply softmax
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            
            # Get probability of "yes"
            scores = batch_scores[:, 1].exp().tolist()
        
        return scores
    
    def rank(
        self, 
        query: str, 
        documents: list[str], 
        instruction: str = None
    ) -> list[float]:
        """
        Rank documents by relevance to query
        
        Args:
            query: Search query
            documents: List of documents to rank
            instruction: Optional task instruction
            
        Returns:
            List of relevance scores (same order as documents)
        """
        # Format all query-document pairs
        pairs = [
            self.format_instruction(query, doc, instruction) 
            for doc in documents
        ]
        
        # Process and score
        inputs = self._process_inputs(pairs)
        scores = self._compute_scores(inputs)
        
        return scores
    
    def rank_with_indices(
        self, 
        query: str, 
        documents: list[str], 
        instruction: str = None
    ) -> list[tuple[int, float]]:
        """
        Rank documents and return sorted indices with scores
        
        Args:
            query: Search query
            documents: List of documents to rank
            instruction: Optional task instruction
            
        Returns:
            List of (index, score) tuples sorted by score (descending)
        """
        scores = self.rank(query, documents, instruction)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked


# Example usage
if __name__ == "__main__":
    # Initialize reranker
    reranker = QwenReranker()
    
    # Example query and documents
    query = "What is the capital of China?"
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "Beijing is a major city in China with a rich history.",
    ]
    
    # Get scores
    scores = reranker.rank(query, documents)
    print("Scores:", scores)
    
    # Get ranked results
    ranked = reranker.rank_with_indices(query, documents)
    print("\nRanked documents:")
    for idx, score in ranked:
        print(f"  [{score:.4f}] {documents[idx][:50]}...")