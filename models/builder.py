from typing import Tuple, List, Dict, Callable, Set
from torch import nn, Tensor
import torch
import itertools


class NamedModule(nn.Module):

    def __init__(self,
                 module: nn.Module,
                 from_names: List[str],
                 to_names: List[str]):
        super().__init__()
        self.module = module
        self.from_names = from_names
        self.to_names = to_names

    def forward(self, name2tensor: Dict[str, Tensor]) -> Dict[str, Tensor]:

        input = [name2tensor[name] for name in self.from_names]

        res = self.module(*input)
        if isinstance(res, Tensor):
            res = [res]

        return dict(zip(self.to_names, res))

    def __call__(self, name2tensor: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return self.forward(name2tensor)


class NamedModuleObject(NamedModule):

    def __init__(self, module: nn.ModuleList,
                 from_names: List[str],
                 to_names: List[str],
                 fwd: Callable[[Dict[str, Tensor]], Dict[str, Tensor]]):
        super().__init__(module, from_names, to_names)
        self.fwd = fwd

    def forward(self, name2tensor: Dict[str, Tensor]):
        return self.fwd(name2tensor)





class OutputMemory:

    def __init__(self):
        self.data: Dict[str, Dict[str, Tensor]] = {}

    def add_output(self, name, data):
        assert name not in self.data
        self.data[name] = data

    def contains_all(self, names: List[str]) -> bool:
        for k in names:
            if k not in self.data:
                return False

        return True

    def filter(self, names: List[str]) -> List[str]:
        return list(filter(lambda k: k not in self.data, names))


class ModuleBuilder:

    def __init__(self):
        self.nodes: Dict[str, NamedModule] = {}
        self.edges: List[Tuple[List[str], str]] = []

    def find_children(self, nodes: Set[str]) -> Set[str]:
        ch = set(map(lambda e: e[1], filter(lambda e: len(set(e[0]) & nodes) > 0, self.edges)))
        return set(filter(lambda n: n not in nodes, ch))

    def are_linked(self, n1: str, n2: str):

        if n1 == n2:
            return True

        dep_edges = [e[0] for e in filter(lambda e: e[1] == name, self.edges)]


    def sub_graph(self, input_nodes: List[str], output: str):
        sub_graph = ModuleBuilder()

        cur_nodes = set(input_nodes)
        new_nodes = self.find_children(cur_nodes)

        while len(new_nodes) > 0:
            cur_nodes += new_nodes
            new_nodes = self.find_children(cur_nodes)

        nodes = dict([(node, self.nodes[node]) for node in cur_nodes])
        edges = filter(lambda e: len(set(e[0]) & cur_nodes) > 0, self.edges)

        sub_graph.nodes = nodes
        sub_graph.edges = edges

        return sub_graph

    def add_module(self, name: str, module: nn.Module, from_names: List[str], to_names: List[str]):
        self.nodes[name] = NamedModule(module, from_names, to_names)

    def add_edge(self, from_names: List[str], to_name: str):
        self.edges.append((from_names, to_name))

    def get_dependent_nodes(self, name: str) -> List[str]:
        dep_edges = [e[0] for e in filter(lambda e: e[1] == name, self.edges)]
        assert len(dep_edges) == 1
        return dep_edges[0]

    def compute(self, name: str, memory: OutputMemory) -> Dict[str, Tensor]:
        if name in memory.data:
            return memory.data[name]
        else:
            deps = self.get_dependent_nodes(name)
            input = {}
            for d in memory.filter(deps):
                res_d = self.compute(d, memory)
                assert len(input.keys() & res_d.keys()) == 0
                input.update(res_d)

            out = self.nodes[name].forward(input)
            memory.add_output(name, out)
            return out

    def build(self, input_nodes: List[str], output_name: str) -> NamedModule:
        sub_builder = self.sub_graph(input_nodes)

        def fwd(input: Dict[str, Tensor]) -> Dict[str, Tensor]:
            memory = OutputMemory()

            for name in input_nodes:
                out = sub_builder.nodes[name].forward(input)
                memory.add_output(name, out)

            return sub_builder.compute(output_name, memory)

        modules = nn.ModuleList(
           map(lambda m: m.module, sub_builder.nodes.values())
        )

        from_names = list(set(itertools.chain(*[self.nodes[node].from_names for node in input_nodes])))
        to_names = self.nodes[output_name].to_names

        return NamedModuleObject(
            modules,
            from_names,
            to_names,
            fwd
        )









