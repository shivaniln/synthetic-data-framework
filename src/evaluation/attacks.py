from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator

class PrivacyAttacks:
    def __init__(self, real, syn, control, n_attacks=50):
        self.real = real
        self.syn = syn
        self.control = control
        # Ensure we don't try to attack more records than we have
        self.n_attacks = min(n_attacks, len(real), len(control))

    def singling_out(self):
        print("Running Singling Out Attack...")
        evaluator = SinglingOutEvaluator(ori=self.real, 
                                         syn=self.syn, 
                                         control=self.control, 
                                         n_attacks=self.n_attacks)
        evaluator.evaluate()
        return evaluator.risk()

    def linkability(self):
        print("Running Linkability Attack...")
        aux_cols = list(self.real.columns)
        evaluator = LinkabilityEvaluator(ori=self.real, 
                                         syn=self.syn, 
                                         control=self.control, 
                                         aux_cols=aux_cols, 
                                         n_attacks=self.n_attacks)
        evaluator.evaluate()
        return evaluator.risk()