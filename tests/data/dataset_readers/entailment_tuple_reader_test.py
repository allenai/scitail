from operator import attrgetter

from allennlp.common.testing import AllenNlpTestCase

from scitail.data.dataset_readers.entailment_tuple_reader import EntailmentTupleReader


class TestEntailmentTupleReader(AllenNlpTestCase):
    def test_read_from_file(self):

        reader = EntailmentTupleReader(max_tokens=200, max_tuples=30)
        dataset = reader.read('tests/fixtures/data/scitail_dgem.tsv')

        instance1 = {"premise": ["geysers", "-", "periodic", "gush", "of", "hot", "water", "at",
                                 "the", "surface", "of", "the", "Earth", "."],
                     "hypothesis": ["The", "surface", "of", "the", "sun", "is", "much", "hotter",
                                    "than", "almost", "anything", "on", "earth", "."],
                     "nodes": [["The", "surface", "of", "the", "sun"],
                               ["is"],
                               ["much", "hotter", "than", "almost", "anything"],
                               ["much", "hotter", "than", "almost", "anything", "on", "earth"],
                               ["earth"]],
                     "edges": ["subj", "subj-obj", "obj", "on"],
                     "label": "neutral"}

        instance2 = {"premise": ["Facts", ":", "Liquid", "water", "droplets", "can", "be",
                                 "changed", "into", "invisible", "water", "vapor", "through", "a",
                                 "process", "called", "evaporation", "."],
                     "hypothesis": ["Evaporation", "is", "responsible", "for", "changing", "liquid",
                                    "water", "into", "water", "vapor", "."],
                     "nodes": [["Evaporation"],
                               ["is"],
                               ["responsible"],
                               ["water", "vapor"],
                               ["changing", "liquid", "water"],
                               ["responsible", "for", "changing", "liquid", "water", "into",
                                "water", "vapor"]],
                     "edges": ["subj", "subj-obj", "obj", "for", "into"],
                     "label": "entails"}
        instance3 = {"premise": ["By", "comparison", ",", "the", "earth", "rotates", "on", "its",
                                 "axis", "once", "per", "day", "and", "revolves", "around", "the",
                                 "sun", "once", "per", "year", "."],
                     "hypothesis": ["Earth", "rotates", "on", "its", "axis", "once", "times", "in",
                                    "one", "day", "."],
                     "nodes": [["Earth"],
                               ["rotates", "on"],
                               ["its", "axis", "once", "times"],
                               ["in"],
                               ["rotates", "on", "its", "axis", "once", "times", "in"],
                               ["one", "day"]],
                     "edges": ["subj", "subj-obj", "obj", "subj", "subj-obj", "obj", "subj",
                               "subj-obj", "obj"],
                     "label": "entails"}
        for instance in dataset.instances:
            fields = instance.fields
            print("\", \"".join([t.text for t in fields["premise"].tokens]))
            print("\", \"".join([t.text for t in fields["hypothesis"].tokens]))
            for edge in fields["edge_labels"].field_list:
                print(edge.label,  end='", "')
        assert len(dataset.instances) == 3
        self.compare_instance(dataset.instances[0].fields, instance1)
        self.compare_instance(dataset.instances[1].fields, instance2)
        self.compare_instance(dataset.instances[2].fields, instance3)

    def compare_instance(self, fields, instance):
        assert [t.text for t in fields["premise"].tokens] == instance["premise"]
        assert [t.text for t in fields["hypothesis"].tokens] == instance["hypothesis"]
        assert fields["label"].label == instance["label"]
        self.compare_nodes(fields, instance)
        self.compare_edges(fields, instance)

    def compare_nodes(self, input_fields, test_instance):
        field_nodes = sorted(input_fields["nodes"].field_list,
                             key=lambda x: [t.text for t in x.tokens])
        instance_nodes = sorted(test_instance["nodes"])
        for node, instance_node in zip(field_nodes, instance_nodes):
            assert [t.text for t in node.tokens] == instance_node

    def compare_edges(self, input_fields, test_instance):
        field_edges = sorted(input_fields["edge_labels"].field_list,
                             key=attrgetter("label"))
        instance_edges = sorted(test_instance["edges"])
        for edge, instance_edge in zip(field_edges, instance_edges):
            assert edge.label == instance_edge
