import time

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset

from collections import defaultdict


class IgEvaluationJob(EvaluationJob):
    """ Predicate classification and detection evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = False

        self.ks = self.config.get("predicate_det_class.ks")
        self.Xs = self.config.get("predicate_det_class.Xs")
        self.max_k = max(self.ks)

        if self.__class__ == IgEvaluationJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        self.triples_class = self.dataset.index(f"{self.eval_split}_triples_class_grouped_ig")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            range(len(self.triples_class)),
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):
        """Looks up true triples for each IG in the batch"""  # Not really what I am planning to do here..
        triples_class = []
        for example_index in batch:
            triples_class.append(self.triples_class[str(example_index)])  # TODO: problem comes from map working on strings despite ints being the better choice

        return {
            "triples_class": triples_class,
            "example_indexes": batch
        }

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )

        S, P, O = 0, 1, 2

        correct_triples = {}
        correct_triples_detection = {}

        for k in self.ks:
            correct_triples[k] = defaultdict(int)
            correct_triples_detection[k] = defaultdict(int)
        total_triples = 0

        # let's go
        epoch_time = -time.time()
        total_time = -time.time()
        score_time = 0
        loop_time = 0
        sorting_etc_time = 0
        finding_time = 0

        columns = torch.arange(self.dataset.num_relations())
        columns_no_unknown = columns[columns != self.dataset.id_unknown()]

        # indexes_to_be_viewed = [18, 35, 78]
        for batch_index, batch in enumerate(self.loader):
            triples_class_batch = batch["triples_class"]
            example_indexes = batch["example_indexes"]

            # TODO: Not sure if I need different ways of transforming to probabilities between
            # TODO: count and IG training

            # Iterate over IGs
            for triples_class, example_index in zip(triples_class_batch, example_indexes):
                # TODO: Probably need to move these tensors (e.g. triples_class) to device, but I could also just use a tensor instead of a list when creating the batch
                """
                if example_index in indexes_to_be_viewed:
                    print("Look here:")
                    print(f"Example Number: {example_index}")
                    print("Triples classes:")
                    print(triples_class)
                    print("Triples instances:")
                    print(triples_instance)
                """
                triples_class = triples_class.to(self.device)
                score_time -= time.time()
                scores = self.model.score_so(triples_class[:, S], triples_class[:, O])
                """
                if example_index in indexes_to_be_viewed:
                    for elem in scores:
                        print(elem)
                """
                score_time += time.time()
                scores = torch.exp(scores)
                probabilities = scores / torch.sum(scores, dim=1).view(-1, 1)

                probabilities = probabilities.to("cpu")



                # probabilities_no_unknown = probabilities[indices_no_unknown]
                """
                if example_index in indexes_to_be_viewed:
                    print("")
                    print("predicted probabilities")
                    for elem in probabilities:
                        print(elem)
                """
                indices_no_unknown = (triples_class[:, P] != (self.dataset.id_unknown())).nonzero().view(-1)
                if self.config.get("train_ig.use_unknown_relation"):
                    probabilities = probabilities[:, columns_no_unknown]

                max_probabilities, max_indices = torch.topk(
                    probabilities, self.max_k, dim=1
                )


                # max_probabilities, max_indices = torch.topk(
                #     probabilities[:, :self.dataset.num_relations() - 1], self.max_k, dim=1
                # )
                """
                if self.config.get("train_ig.use_unknown_relation"):
                    indices_no_unknown = (triples_class[:, P] != (self.dataset.id_unknown())).nonzero().view(-1)
                    max_probabilities, max_indices = torch.topk(
                        probabilities[:, :self.dataset.num_relations() - 1], self.max_k, dim=1
                    )
                else:
                    indices_no_unknown = (triples_class[:, P] != (self.dataset.num_relations())).nonzero().view(-1)
                    max_probabilities, max_indices = torch.topk(
                        probabilities[:, :self.dataset.num_relations()], self.max_k, dim=1
                    )
                """

                """
                if example_index in indexes_to_be_viewed:
                    print("maximum predicted probabilities")
                    print(max_probabilities)
                    print(max_indices)
                """

                # max_indices = max_indices.type(torch.IntTensor)
                true_predicates = triples_class[:, P].reshape(-1, 1).to("cpu")
                true_predicates = true_predicates.repeat(1, self.max_k)



                """
                if example_index in indexes_to_be_viewed:
                    print("true predicates")
                    print(true_predicates)
                """

                # max_indices = true_predicates  # to test if calculation works correctly

                true_predictions = torch.eq(true_predicates, max_indices)  # matrix where all correct predictions are marked as True

                """
                if example_index in indexes_to_be_viewed:
                    print("true predictions")
                    print(true_predictions)
                """

                true_predictions_indexes = [
                    (row[0] + row[1] * len(triples_class)).item()
                    for row in true_predictions.nonzero()
                ]
                true_predictions_indexes.sort()

                true_predictions_no_unknown = true_predictions[indices_no_unknown]
                true_predictions_no_unknown_indexes = [
                    (row[0] + row[1] * len(indices_no_unknown)).item()
                    for row in true_predictions_no_unknown.nonzero()
                ]
                true_predictions_no_unknown_indexes.sort()

                """
                if example_index in indexes_to_be_viewed:
                    print("true prediction indexes")
                    print(true_predictions_indexes)
                """

                loop_time -= time.time()
                for k in self.ks:
                    sorting_etc_time -= time.time()
                    max_probabilities_k = max_probabilities[:, :k].transpose(dim0=1, dim1=0).reshape(-1)
                    sorted_indices = torch.sort(max_probabilities_k, dim=0, descending=True)[1]
                    max_probabilities_k_no_unknown = max_probabilities[indices_no_unknown, :k].transpose(dim0=1, dim1=0).reshape(-1)
                    sorted_indices_no_unknown = torch.sort(max_probabilities_k_no_unknown, dim=0, descending=True)[1]
                    sorting_etc_time += time.time()
                    finding_time -= time.time()
                    for index in true_predictions_indexes:
                        if index >= (k * len(triples_class)):
                            break
                        rank = (sorted_indices == index).nonzero().item()
                        correct_triples[k][rank] += 1

                    for index in true_predictions_no_unknown_indexes:
                        if index >= (k * len(indices_no_unknown)):
                            break
                        rank = (sorted_indices_no_unknown == index).nonzero().item()
                        correct_triples_detection[k][rank] += 1

                    finding_time += time.time()

                loop_time += time.time()
                total_triples += len(indices_no_unknown)

                if self.trace_examples:
                    1+1
                    # TODO: Write code for example tracing

            if self.trace_batch:
                1+1
                # TODO: Write code for batch tracing

        metrics = {}

        metrics_calculation_time = -time.time()
        for k in self.ks:
            correct_triples_cum = torch.zeros(100)
            correct_triples_detection_cum = torch.zeros(100)
            correct_triples_cum[0] = correct_triples[k][0]
            correct_triples_detection_cum[0] = correct_triples_detection[k][0]
            for i in range(1, 100):
                correct_triples_cum[i] = correct_triples[k][i] + correct_triples_cum[i - 1]
                correct_triples_detection_cum[i] = correct_triples_detection[k][i] + correct_triples_detection_cum[i - 1]

            for at_x in self.Xs:
                metrics[f"predicate_classification_recall_at_{at_x}_k={k}"] = \
                    (correct_triples_cum[at_x - 1] / total_triples).item()
                metrics[f"predicate_detection_recall_at_{at_x}_k={k}"] = \
                    (correct_triples_detection_cum[at_x - 1] / total_triples).item()

        metrics_calculation_time += time.time()

        total_time += time.time()


        print(f"Loop Time: {round(loop_time, 3)}")
        print(f"Score Time: {round(score_time, 3)}")
        print(f"Other Time: {round(total_time, 3)}")

        print(f"Sorting etc. Time: {round(sorting_etc_time, 3)}")

        print(f"Finding Time: {round(finding_time, 3)}")
        print(f"Metrics calculation time: {round(metrics_calculation_time, 3)}")

        # must return some trace entry in the form of a dict which has at
        # least a value for some "metric_name" which is used to update lr

        # TODO: I basically have the results but now I need to put them in whatever kind of
        # TODO: structure they have to be in for this framework to work

        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="ig_evaluation",
            scope="epoch",
            split=self.eval_split,
            # filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples_class),
            epoch_time=epoch_time,
            event="eval_completed",
            **metrics,
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **trace_entry),
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        for f in self.post_valid_hooks:
            f(self, trace_entry)

        return trace_entry
