import torch
import time
import kge.job
from kge.job import EvaluationJob


class EntityPairRankingJob(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.is_prepared = False

    def _prepare(self):
        """Create dataloader and construct all indexes needed to run."""

        if self.is_prepared:
            return

        # Create indexes and evaluation data
        # These indexes hold a list of (s,o) tuples for every relation p
        self.train_p = self.dataset.index_MtoN("train")
        self.valid_p = self.dataset.index_MtoN("valid")
        self.test_p = self.dataset.index_MtoN("test")

        # Get set of relations in data
        if self.eval_data == "test":
            self.triples = self.dataset.test
            self.eval_index = self.test_p
        else:
            self.triples = self.dataset.valid
            self.eval_index = self.valid_p
        self.relations = torch.unique(self.triples[:, 1])

        # Create data loader
        # TODO for now hardcoding batch size to 1
        self.loader = torch.utils.data.DataLoader(
            self.relations,
            # collate_fn=self._collate,
            shuffle=False,
            batch_size=1,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )

        # Let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    # TODO check correct use of device throughout this code
    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on " + self.eval_data + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities

        # These histograms will gather information about the ranks of the
        # correct answer (raw rank and filtered rank)
        # TODO for PR I think these should be size k because we only sort the top k
        # Confirm when implementing
        hist = torch.zeros([self.max_k], device=self.device, dtype=torch.float)
        hist_filt = torch.zeros([self.max_k], device=self.device, dtype=torch.float)

        topk_scores = {}
        topk_indices = {}

        epoch_time = -time.time()
        for batch_number, batch in enumerate(self.loader):
            # construct a label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            relation = batch.item()
            # TODO handle filtering
            # filtering_coords = batch_coords[1].to(self.device)
            # test_triples_coords = batch_coords[2].to(self.device)
            # labels = kge.job.util.coord_to_sparse_tensor(
            #     num_entities,
            #     num_entities,
            #     filtering_coords,
            #     self.device,
            #     float("Inf"),
            #     depth=batch_size
            # ).to_dense()

            # compute all scores
            # Assumes batch_size = 1 for now
            topk_scores[relation], topk_indices[relation] = self._topk_scores(batch)

            # compute metrics
            hits, map = self._compute_metrics(topk_indices[relation], self.eval_index[relation])

            print(relation, hits, map)

            # TODO add tracing and logging

        # we are done; compute final metrics
        print("\033[2K\r", end="", flush=True)  # clear line and go back
        metrics = self._compute_metrics(hist)
        metrics.update(self._compute_metrics(hist_filt, suffix="_filtered"))
        epoch_time += time.time()

        # TODO add tracing and logging

        return trace_entry

    def _topk_scores(self, p, filter_index=None):
        r"""Returns top k scores and corresponding indices of entire score matrix
         of given p. If filter_index is given, also returns filtered top k scores
         and indices."""

        # TODO: current implementation does not prevent inverse relations model
        # from being used, because we rely on scores_po here
        # But this should be prevented somewhere

        # Get chunk size to divide computation of score matrix
        chunk_size = self.config.get_default("entity_pair_ranking.chunk_size")
        num_chunks = int(self.dataset.num_entities / chunk_size)

        # Initialize scores tensor (3-way: depth, rows, cols)
        all_topk_scores = torch.Tensor()
        all_topk_indices = torch.Tensor().int()

        # Compute score matrix in chunks
        for chunk in range(num_chunks + 1):
            start = chunk * chunk_size
            end = start + chunk_size
            if end > self.dataset.num_entities:
                end = self.dataset.num_entities
            o = torch.Tensor(list(range(start, end)))
            p_exp = p.expand(len(o))

            # Compute chunk scores
            # Shape is (chunk_size, num_entities), each column for a subject
            chunk_scores = self.model.score_po(p_exp, o)

            # TODO apply filtering to chunk scores
            # Then do the rest also with chunk_scores_filt

            # Get top k scores in chunk which are already sorted by torch.topk
            # Transposing for correct indexing, we need (s, o)
            # TODO making a copy with contiguous not cool!
            chunk_scores_1d = chunk_scores.t().contiguous().view(1, -1)
            topk = torch.topk(chunk_scores_1d, k=self.max_k)

            # Convert topk.indices for current chunk to global 2D indices of score matrix
            topk_indices_2d = kge.job.util.convert_1d_indices_to_2d(topk.indices,
                                                                    len(o),
                                                                    chunk_size * chunk)

            # Add (topk.scores, global_2d_indices) to global tracking
            all_topk_scores = torch.cat((all_topk_scores, topk.values[0,:]))
            all_topk_indices = torch.cat((all_topk_indices, topk_indices_2d))

        # Get global top k scores
        global_topk = torch.topk(all_topk_scores, k=self.max_k)

        # Return global scores and corresponding indices
        return global_topk.values, all_topk_indices[global_topk.indices, :]

    def _compute_metrics(self, topk_indices, test_triples_coords):

        # Find matches between topk_indices[relation] and test_triples_coords[relation]
        # TODO need an efficient join
        # Elementwise operations on sparse tensors would helpa
        # Should I just do a hash join here?
        hits_at_k = 0
        map_at_k = 0
        for topk_index in topk_indices:
            for test_triple in test_triples_coords:
                if torch.equal(topk_index, test_triple):
                    hits_at_k += 1

        # TODO compute MAP, topk_indices already come sorted
        # TODO add weights: min(self.max_k, len(test_triples_coords)) / total for all relations

        return hits_at_k, map_at_k