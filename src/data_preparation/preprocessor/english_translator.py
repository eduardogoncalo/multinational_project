from sklearn.base import BaseEstimator, TransformerMixin
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline, MarianMTModel, MarianTokenizer
import pandas as pd

class EnglishTranslator(BaseEstimator, TransformerMixin):

    def __init__(self, use_case=2):
        self.use_case = use_case
        
        
        self.t2t_m = "Helsinki-NLP/opus-mt-mul-en"
        
        
        self.model = MarianMTModel.from_pretrained(self.t2t_m)
        self.tokenizer = MarianTokenizer.from_pretrained(self.t2t_m)
        
        if self.use_case == 1:
            self.t2t_pipe = pipeline("translation", model=self.model, tokenizer=self.tokenizer)
        else:
            self.t2t_pipe = None
        
        self.nlp_stanza = stanza.Pipeline(
            lang="multilingual", 
            processors="langid",
            download_method=DownloadMethod.REUSE_RESOURCES
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].astype(str).tolist()
        elif isinstance(X, pd.Series):
            texts = X.astype(str).tolist()
        else:
            texts = [str(text) for text in list(X)]
            
        return self._translate_to_en(texts)

    def _translate_to_en(self, texts: list) -> list[str]:
        text_en_l = []
        for text in texts:
            
            if pd.isna(text):
                text = ""
            else:
                text = str(text)

            if not text or text.strip() == "" or text.strip().lower() == "nan":
                text_en_l.append("")
                continue

            doc = self.nlp_stanza(text)
            lang = doc.lang
            
            if lang == "en":
                text_en_l.append(text)
            else:
                
                if self.use_case == 1:
                    text_en = self.t2t_pipe(
                        text, 
                        truncation=True,
                        max_length=512
                    )
                    text_en = text_en[0]['translation_text']
                    
                elif self.use_case == 2:
                    encoded_hi = self.tokenizer(
                        text, 
                        return_tensors="pt", 
                        truncation=True,     
                        max_length=512      
                    )
                    
                    generated_tokens = self.model.generate(**encoded_hi)
                    text_en = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                else:
                    text_en = text
                
                text_en_l.append(text_en)
                
        return text_en_l