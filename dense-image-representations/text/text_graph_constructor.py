import torch
from torch_geometric.data import Data as GraphData
from transformers import AutoTokenizer, T5EncoderModel

import glob
import json 
import re
import pdb
import os 

class TextGraphConstructor:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = T5EncoderModel.from_pretrained("t5-small")
        self.model.to("cuda")

    def encode_with_lm(self, texts):
        """Accepts a list of texts and returns a list of encoded texts."""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids
        inputs = inputs.to("cuda")
        outputs = self.model(inputs).last_hidden_state
        return outputs.mean(dim=1)

    def __call__(self, parsed_caption):
        """Accepts a parsed caption and returns a graph as an instance of torch_geometric.data.Data.

        Parsed caption must be a dictionary with keys 'objects' and 'relationships'.
        'objects' is a list of strings, each string representing an object.
        'relationships' is a list of lists, each sub list contains 3 items, the first two of
        which are indices of the objects and the third item is a string representing the relationship
        between the two objects.
        """
        nodes = self.encode_with_lm(parsed_caption["objects"])
        relationships = parsed_caption["relationships"]
        edge_index = torch.tensor(list(map(lambda x: x[:2], relationships))).T
        if len(relationships) > 0:
            edge_names = list(map(lambda x: x[2], relationships))
            edge_attr = self.encode_with_lm(list(map(lambda x: x[2], relationships)))
            graph = GraphData(x=nodes, edge_index=edge_index, edge_attr=edge_attr, edge_names=edge_names, node_names=parsed_caption['objects'])
        else:
            graph = GraphData(x=nodes, edge_index=edge_index, node_names=parsed_caption['objects'])
        return graph


if __name__ == "__main__":
    text_graph_constructor = TextGraphConstructor()
    # parsed_caption = {
    #     "objects": ["dog", "ball"],
    #     "relationships": [[0, 1, "playing with"]],
    # }
    # graph = text_graph_constructor(parsed_caption)
    # print(
    #     graph.num_nodes,
    #     graph.is_directed(),
    #     graph.num_edges,
    #     graph.num_node_features,
    #     graph.num_edge_features,
    # )

    l = glob.glob('../coco_parsed_captions/*.json')
    for f in l:
        if not os.path.exists('../coco_text_graph/{cap_id}.pt'):
            with open(f, 'r') as file:
                parsed_caption = json.load(file)
            graph = text_graph_constructor(parsed_caption)
            cap_id = re.findall('coco_parsed_captions\/(.*)\.json', f)[0]
            torch.save(graph, f'../coco_text_graph/{cap_id}.pt')

            