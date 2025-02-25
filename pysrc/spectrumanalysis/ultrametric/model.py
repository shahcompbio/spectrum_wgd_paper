import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.poutine import trace
from pyro.nn import PyroModule
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class UltrametricModel(PyroModule):
    def __init__(self, branchlength_matrix, opportunities, branches, patient_age):
        super(UltrametricModel, self).__init__()
        self.branchlength_matrix = branchlength_matrix
        self.opportunities = opportunities
        self.branches = branches
        self.patient_age = patient_age

    def model(self, snv_counts=None):
        n_orderings = self.branchlength_matrix.shape[0]
        n_branches = self.branchlength_matrix.shape[1]
        n_branch_starts = self.branchlength_matrix.shape[2]

        assert self.opportunities.shape[0] == n_branches

        # Uniformly sample an ordering of the branches
        ordering = pyro.sample('ordering', dist.Categorical(torch.ones([n_orderings])/n_orderings))

        # Uniformly sample an ordering of the branches
        branch_starts = pyro.sample('branch_starts', dist.Dirichlet(torch.ones([n_branch_starts])))

        # Calculate branch lengths
        # Flexible handling of enumerated dimension
        # Check if there's an extra dimension due to enumeration
        if len(ordering.shape) > 0:
            # Adjust for the extra enumeration dimension
            expanded_branch_starts = branch_starts.unsqueeze(0).expand(ordering.shape[0], -1)
            branch_lengths = torch.bmm(self.branchlength_matrix[ordering], expanded_branch_starts.unsqueeze(-1)).squeeze(-1)
        else:
            # Normal execution without extra dimension
            branch_lengths = torch.mv(self.branchlength_matrix[ordering], branch_starts)

        # Track branch lengths for posterior estimation
        pyro.deterministic('branch_lengths', branch_lengths)

        # Track branch lengths in years
        pyro.deterministic('branch_lengths_years', branch_lengths * self.patient_age)

        # Random mutation rate with prior from gerstung et al.
        mutrate_shape = 8.595284736381526
        mutrate_rate = 0.10713681682913984
        mutation_rate = pyro.sample('mutation_rate', dist.Gamma(mutrate_shape, mutrate_rate))

        # Random acceleration with prior from gerstung et al.
        acc_shape = 17.72240526422662
        acc_rate = 0.3508012972631267
        mut_rate_accel = pyro.sample('mut_rate_accel', dist.Gamma(acc_shape, acc_rate))
        post_malignant_rate = mutation_rate * mut_rate_accel

        # Random interval for non-accelerated rate on root branch
        pre_malignant_interval = pyro.sample('pre_malignant_interval', dist.Beta(1, 1))

        # Track pre-malignant interval in years, assume the first branch is the root
        pyro.deterministic('pre_malignant_interval_years', pre_malignant_interval * branch_lengths[0] * self.patient_age)

        # Total mutation rate of root branch
        root_branch_rate = mutation_rate * pre_malignant_interval + post_malignant_rate * (1. - pre_malignant_interval)

        # Mutation rates across all branches
        branch_mutation_rate = torch.cat([root_branch_rate.unsqueeze(0), post_malignant_rate.repeat(n_branches - 1)])

        # Track branch_mutation_rate for posterior estimation
        pyro.deterministic('branch_mutation_rate', branch_mutation_rate)

        # Expected SNV count
        expected_count = branch_lengths * branch_mutation_rate * self.opportunities * self.patient_age

        # Track expected_count for posterior estimation
        pyro.deterministic('expected_count', expected_count)

        pyro.sample(f'snv_counts', dist.Poisson(rate=expected_count).to_event(), obs=snv_counts)

        return ordering, branch_starts, branch_lengths, pre_malignant_interval

    def fit(self, snv_counts=None, condition_vars=(), num_steps=1000):
        condition_vars = {k: torch.tensor(v) for k, v in dict(condition_vars).items()}

        self.conditioned_model = config_enumerate(pyro.condition(
            self.model,
            data=condition_vars,
        ))

        optim = pyro.optim.Adam({'lr': 0.1, 'betas': [0.8, 0.99]})
        elbo = TraceEnum_ELBO(max_plate_nesting=0)

        pyro.set_rng_seed(1)
        pyro.clear_param_store()

        guide = AutoNormal(
            pyro.poutine.block(self.conditioned_model, expose=[
                'branch_starts',
                'mutation_rate',
                'mut_rate_accel',
                'pre_malignant_interval']))

        self.svi = SVI(self.conditioned_model, guide, optim, loss=elbo)

        self.losses = []
        for i in range(num_steps):
            loss = self.svi.step(snv_counts=snv_counts)
            self.losses.append(loss)

    def sample_posterior(self, num_samples=1000):
        posterior_predictive = pyro.infer.Predictive(self.conditioned_model, guide=self.svi.guide, num_samples=num_samples)

        samples = posterior_predictive()

        for var in samples:
            samples[var] = samples[var].detach().numpy()
            
        return samples
    
    def posterior_mode(self, snv_counts, num_posterior_samples=1000):
        samples = self.sample_posterior(num_samples=num_posterior_samples)
        
        mutation_rate = torch.tensor(np.median(samples['mutation_rate']))
        mut_rate_accel = torch.tensor(np.median(samples['mut_rate_accel']))
        pre_malignant_interval = torch.tensor(np.median(samples['pre_malignant_interval']))
        branch_starts = torch.tensor(np.median(samples['branch_starts'], axis=0))
        branch_starts = branch_starts / branch_starts.sum()
        
        conditioned_model = pyro.condition(
            self.model,
            data={
                'mutation_rate': mutation_rate,
                'mut_rate_accel': mut_rate_accel,
                'pre_malignant_interval': pre_malignant_interval,
                'branch_starts': branch_starts,
            },
        )

        traced_model = trace(conditioned_model).get_trace(snv_counts=snv_counts)

        posterior_info = {
            'model_log_prob': traced_model.log_prob_sum().detach().numpy(),
            'log_probs': {},
            'values': {},
            'branches': self.branches,
        }

        for name, node in traced_model.nodes.items():
            if node["type"] == "sample":
                posterior_info['log_probs'][name] = node['log_prob_sum'].detach().numpy()
                posterior_info['values'][name] = node['value'].detach().numpy()

        return posterior_info

