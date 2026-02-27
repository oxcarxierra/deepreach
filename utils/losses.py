import torch

# uses real units
def init_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brt_hjivi_loss(state, value, dvdt, dvds, boundary_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, value - boundary_value)
        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                # pretraining
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}

    return brt_hjivi_loss
def init_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    def brat_hjivi_loss(state, value, dvdt, dvds, boundary_value, reach_value, avoid_value, dirichlet_mask, output):
        if torch.all(dirichlet_mask):
            # pretraining loss
            diff_constraint_hom = torch.Tensor([0])
        else:
            ham = dynamics.hamiltonian(state, dvds)
            if minWith == 'zero':
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = dvdt - ham
            if minWith == 'target':
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, value - reach_value), value + avoid_value)

        dirichlet = value[dirichlet_mask] - boundary_value[dirichlet_mask]
        if dynamics.deepreach_model == 'exact':
            if torch.all(dirichlet_mask):
                dirichlet = output.squeeze(dim=-1)[dirichlet_mask]-0.0
            else:
                return {'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
        return {'dirichlet': torch.abs(dirichlet).sum() / dirichlet_loss_divisor,
                'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}
    return brat_hjivi_loss


def init_transreach_brt_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    """Sequential HJI loss for TransReach (BRT setting)."""

    def transreach_brt_hjivi_loss(
        seq_state,
        seq_value,
        seq_dvdt,
        seq_dvds,
        seq_boundary_value,
        seq_dirichlet_mask,
        seq_output,
    ):
        if torch.all(seq_dirichlet_mask):
            diff_constraint_hom = torch.zeros(
                1, device=seq_value.device, dtype=seq_value.dtype
            )
        else:
            ham = dynamics.hamiltonian(seq_state, seq_dvds)
            if minWith == "zero":
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = seq_dvdt - ham
            if minWith == "target":
                diff_constraint_hom = torch.max(
                    diff_constraint_hom, seq_value - seq_boundary_value
                )
            diff_constraint_hom = diff_constraint_hom[~seq_dirichlet_mask]

        dirichlet = seq_value[seq_dirichlet_mask] - seq_boundary_value[seq_dirichlet_mask]
        if dynamics.deepreach_model == "exact":
            if torch.all(seq_dirichlet_mask):
                dirichlet = seq_output.squeeze(dim=-1)[seq_dirichlet_mask] - 0.0
            else:
                return {"diff_constraint_hom": torch.abs(diff_constraint_hom).mean()}

        return {
            "dirichlet": torch.abs(dirichlet).mean() / dirichlet_loss_divisor,
            "diff_constraint_hom": torch.abs(diff_constraint_hom).mean(),
        }

    return transreach_brt_hjivi_loss


def init_transreach_brat_hjivi_loss(dynamics, minWith, dirichlet_loss_divisor):
    """Sequential HJI-VI loss for TransReach (BRAT setting)."""

    def transreach_brat_hjivi_loss(
        seq_state,
        seq_value,
        seq_dvdt,
        seq_dvds,
        seq_boundary_value,
        seq_reach_value,
        seq_avoid_value,
        seq_dirichlet_mask,
        seq_output,
    ):
        if torch.all(seq_dirichlet_mask):
            diff_constraint_hom = torch.zeros(
                1, device=seq_value.device, dtype=seq_value.dtype
            )
        else:
            ham = dynamics.hamiltonian(seq_state, seq_dvds)
            if minWith == "zero":
                ham = torch.clamp(ham, max=0.0)

            diff_constraint_hom = seq_dvdt - ham
            if minWith == "target":
                diff_constraint_hom = torch.min(
                    torch.max(diff_constraint_hom, seq_value - seq_reach_value),
                    seq_value + seq_avoid_value,
                )
            diff_constraint_hom = diff_constraint_hom[~seq_dirichlet_mask]

        dirichlet = seq_value[seq_dirichlet_mask] - seq_boundary_value[seq_dirichlet_mask]
        if dynamics.deepreach_model == "exact":
            if torch.all(seq_dirichlet_mask):
                dirichlet = seq_output.squeeze(dim=-1)[seq_dirichlet_mask] - 0.0
            else:
                return {"diff_constraint_hom": torch.abs(diff_constraint_hom).mean()}

        return {
            "dirichlet": torch.abs(dirichlet).mean() / dirichlet_loss_divisor,
            "diff_constraint_hom": torch.abs(diff_constraint_hom).mean(),
        }

    return transreach_brat_hjivi_loss
