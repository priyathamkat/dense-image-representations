from typing import List, Optional

from absl import flags, app
import json

from llama import Llama, Dialog

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ckpt_dir",
    "/cmlscratch/pkattaki/developer/llama/llama-2-7b-chat/",
    "Path to the Llama 2 chat model checkpoint directory.",
)
flags.DEFINE_string(
    "tokenizer_path",
    "/cmlscratch/pkattaki/developer/llama/tokenizer.model",
    "Path to the Llama 2 chat model tokenizer.",
)


class CaptionParser:
    def __init__(
        self,
        max_seq_len: int = 512,
        max_batch_size: int = 8,
    ):
        self.generator = Llama.build(
            ckpt_dir=FLAGS.ckpt_dir,
            tokenizer_path=FLAGS.tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

    def ask_llama(
        self, captions: List[str], temperature=0.6, top_p=0.9, max_gen_len=None
    ) -> List[str]:
        """
        Use Llama 2 to parse the captions.

        Args:
            captions (List[str]): The input captions to be parsed.
            temperature (float, optional): The temperature value for controlling randomness in generation.
                Defaults to 0.6.
            top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
                Defaults to 0.9.
            max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
                set to the model's max sequence length. Defaults to None.
        """
        dialogs: List[Dialog] = list(
            [
                {
                    "role": "system",
                    "content": """You are a helpful assistant who is designed to parse sentences.
                You are required to identify the objects mentioned in the sentences, their attributes and relationships between them.
                """,
                },
                {
                    "role": "user",
                    "content": f"""In the following sentence:
                    a tennis player in all white swinging at the ball
                list the objects with their attributes and relationships between them.
                Output should be in JSON format with fields 'objects', 'relationships'
                Each relationship is a list of the form [subject, object, predicate].""",
                },
                {
                    "role": "assistant",
                    "content": """
                    {
                        "objects":["tennis player in all white", "ball"],
                        "relationships":[["tennis player", "ball", "swinging at"]]
                    }
                """,
                },
                {
                    "role": "user",
                    "content": f"""Parse the following sentence:
                    {caption}
                and output just a single JSON object in the same format as above without any explanation.""",
                },
            ]
            for caption in captions
        )

        results = self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        results = [result["generation"]["content"] for result in results]
        return results

    def parse(self, captions: List[str]) -> List[dict[str, str]]:
        """
        Parse the captions to extract objects, attributes and the relationships between them.

        Args:
            captions (List[str]): The input captions to be parsed.

        Returns:
            List[dict[str, str]]: The parsed captions in the following format:
                {
                    "objects": List[str],
                    "relationships": List[List[str, str, str]]
                }
        """
        results = self.ask_llama(captions)
        outputs = []
        for result in results:
            try:
                decoded_result = json.loads(result)
            except json.decoder.JSONDecodeError as e:
                decoded_result = {"objects": [], "relationships": []}
            finally:
                outputs.append(decoded_result)
        return outputs


def main(_):
    parser = CaptionParser()
    print(
        parser.parse(
            captions=[
                "the dog is running on the field.",
                "the cat is sitting on the mat.",
                "three people sit at a table holding lollipops.",
                "a slice of pepperoni pizza on a piece of parchment paper on a paper plate.",
                "tennis player in all white swinging at a ball.",
            ],
        )
    )


if __name__ == "__main__":
    app.run(main)
