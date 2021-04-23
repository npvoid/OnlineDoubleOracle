import numpy as np
from policies.expert_mode_policies import ExpertModeMultiplicativeWeights
import itertools
from games.leduc_poker import calc_ev, get_exploitability, leduc_reset, LEDUC_CARDS, LEDUC_ACTIONS, LeducPoker


class LeducMetagame:
    def __init__(self, pop_size1=1, pop_size2=1, seed=0):
        self.env = LeducPoker()
        self.pop_size1 = pop_size1
        self.pop_size2 = pop_size2
        self.pop1 = LeducPop(self.env, self.pop_size1, seed=seed, skip_illegal=True)
        self.pop2 = LeducPop(self.env, self.pop_size2, seed=seed+int(1e6), skip_illegal=True)
        self.get_payoff = self.pop1.get_payoff  # pop1 is row & pop2 is col

        # Metagame
        self.metagame = np.zeros((self.pop1.pop_size, self.pop2.pop_size))
        for i, hagent1 in enumerate(self.pop1.hpop):
            for j, hagent2 in enumerate(self.pop2.hpop):
                self.metagame[i, j] = self.get_payoff(hagent1, hagent2)

    def update_metagame(self, row_size, col_size):
        # A new BR has been added, we update the metagame
        if (self.metagame.shape[0]==row_size) and (self.metagame.shape[1]==col_size):
            return self.metagame
        self.pop_size1 = row_size
        self.pop_size2 = col_size
        new_metagame = np.zeros((self.pop_size1, self.pop_size2))
        for i, hagent1 in enumerate(self.pop1.hpop[:self.pop_size1]):
            for j, hagent2 in enumerate(self.pop2.hpop[:self.pop_size2]):
                val = self.metagame[i, j] if (i<self.metagame.shape[0] and j<self.metagame.shape[1]) else np.NaN
                new_metagame[i, j] = val if not np.isnan(val) else self.get_payoff(hagent1, hagent2)
        self.metagame = new_metagame
        return self.metagame



class LeducPop:
    def __init__(self, env, init_pop_size, seed=0, skip_illegal=False):

        # Environment
        self.env = env
        self.infostates = self.env.infostates
        self.illegal_action_dict = self.env.illegal_action_dict
        self.skip_illegal = skip_illegal
        assert len(self.infostates) == 936

        # Population
        self.pop_size = init_pop_size
        self.hpop = [self.new_agent(seed=seed + i) for i in range(self.pop_size)]  # key->prob
        self.vpop = [self.agent2vec(agent) for agent in self.hpop]  # Vector population
        self.kpop = [self.vec2key(avec) for avec in self.vpop]  # Key population

        # Policies to aggregate
        self.policies_to_aggregate = None

    def remove_illegal(self, hagent):
        for key, mask in self.illegal_action_dict.items():
            hagent[key] *= mask
            hagent[key] /= np.sum(hagent[key]) + 1e-40
        return hagent

    def new_agent(self, seed=0):
        r = np.random.RandomState(seed)
        hagent = {}
        for infostate in self.infostates:
            hagent[infostate] = np.zeros(3)
            valid_actions = np.nonzero(self.illegal_action_dict[infostate])[0]
            a_idx = r.choice(valid_actions, 1)[0]
            hagent[infostate][a_idx] = 1.
        return hagent

    def agent2vec(self, agent):
        avec = np.zeros((len(self.infostates), 3))
        for i, infostate in enumerate(self.infostates):
            avec[i, :] = agent[infostate]
        return avec

    def vec2agent(self, avec):
        hagent = {}
        for i, infostate in enumerate(self.infostates):
            hagent[infostate] = avec[i, :]
        return hagent

    def vec2key(self, avec):
        action_list = np.nonzero(avec)[1].tolist()
        assert len(action_list) == len(self.infostates), 'A policy was not a pure strategy'
        key = ''.join(str(a) for a in action_list)
        return key

    def hagent_in_pop(self, new_hagent):
        key = self.vec2key(self.agent2vec(new_hagent))
        return key in self.kpop

    def add_agent(self, br_hagent):
        self.hpop.append(br_hagent)
        self.vpop.append(self.agent2vec(self.hpop[-1]))
        self.kpop.append(self.vec2key(self.vpop[-1]))
        self.pop_size += 1

    def get_payoff(self, hagent1, hagent2):
        # Payoff of hagent1
        payoff = 0.
        for cards in itertools.permutations(LEDUC_CARDS, 2):
            cards = list(cards)
            history, stakes, ante, current_player = leduc_reset()
            v1_0, _ = calc_ev(hagent1, hagent2, cards[:], history, stakes, ante[:], current_player, skip_illegal=self.skip_illegal)  # hagent1 as player 0
            _, v1_1 = calc_ev(hagent2, hagent1, cards[:], history, stakes, ante[:], current_player, skip_illegal=self.skip_illegal)  # hagent1 as player 1
            payoff += v1_0+v1_1
        return payoff / 30.

    def agg_agents(self, weights):

        self.policies_to_aggregate = self.hpop[:len(weights)]
        agg_agent = {}
        for agg_player in range(2):
            for cards in itertools.permutations(LEDUC_CARDS, 2):
                cards = list(cards)
                history, stakes, ante, current_player = leduc_reset()
                self.rec_aggregate(agg_agent, weights[:], agg_player, cards, history, stakes, ante[:], current_player)

        # Normalise
        for key, val in agg_agent.items():
            denom = np.sum(val) + 1e-40
            agg_agent[key] /= denom
        if self.skip_illegal:
            agg_agent = self.remove_illegal(agg_agent)
        return agg_agent

    def rec_aggregate(self, agg_agent, agg_reaches, agg_player, cards, history, stakes, ante, current_player):

        # Return payoff is terminal node
        if LeducPoker.is_terminal(history):
            return

        # Chance node
        if len(history) and history[-1] == '.' and history[-2:] != '..':
            deck = [card for card in LEDUC_CARDS if card not in cards]
            for idx, common_card in enumerate(deck):
                new_history = history + '.'
                new_cards = cards + [common_card]
                self.rec_aggregate(agg_agent, agg_reaches, agg_player, new_cards, new_history, stakes, ante[:], current_player)

        # Player node
        else:
            # Compute infostate key
            round = history.count('..') + 1
            key = cards[current_player] + history if round == 1 else cards[current_player] + cards[2] + history
            next_player = 1 - current_player

            probs = [pol[key] for pol in self.policies_to_aggregate]
            if agg_player==current_player:
                if key not in agg_agent:
                    agg_agent[key] = np.array([0., 0., 0.])

            for a_idx, a in enumerate(LEDUC_ACTIONS):
                round_history = history.split('..')[-1]
                if self.skip_illegal and self.env.action_is_illegal(a, round_history, stakes, ante[:], current_player):
                    continue
                new_agg_reaches = np.copy(agg_reaches)
                if agg_player==current_player:
                    for i in range(len(probs)):
                        # compute the new reach for each policy for this action
                        new_agg_reaches[i] *= probs[i][a_idx]
                        # add reach * prob(a) for this policy to the computed policy
                        agg_agent[key][a_idx] += new_agg_reaches[i]

                new_history, new_stakes, new_ante = LeducPoker.apply_action(history, stakes, ante[:], a, current_player)
                self.rec_aggregate(agg_agent, new_agg_reaches, agg_player, cards, new_history, new_stakes, new_ante[:],
                                   next_player)



def re_init_no_regret(current_algo, current_strategy_set, eps, current_avg_policy, new_strategy_set, best_response):
    assert current_algo.n+1 == len(new_strategy_set)
    new_algo = current_algo.__class__(current_algo.n+1, eps)
    index_for_old_actions = np.arange(new_algo.n)[new_strategy_set != best_response]
    index_for_best_response = np.arange(new_algo.n)[new_strategy_set == best_response]

    new_weights = np.zeros(new_algo.n)
    assert np.isclose(np.sum(current_avg_policy), 1.0)
    new_weights[index_for_old_actions] = (current_algo.n / new_algo.n) * current_avg_policy[current_strategy_set]
    new_weights[index_for_best_response] = 1/new_algo.n

    new_algo.set_weights(new_weights)
    return new_algo


class OnlineOracleLeduc(object):
    def __init__(self, leduc_pop: LeducPop, epsilon: float, whole_time_average=True, ref_func=None):
        self.leduc_pop = leduc_pop
        self.pop_size = self.leduc_pop.pop_size
        self.lr = epsilon
        self.whole_time_average = whole_time_average
        self.exp = None
        self.exps = []

        self.current_strategies = list(np.arange(self.pop_size))
        self.new_strategies = self.current_strategies[:]
        self.losses = np.zeros(self.pop_size)
        self.algo = ExpertModeMultiplicativeWeights(len(self.current_strategies), epsilon)
        self.num_intervals = 0
        self.count = 0
        self.update_count = 0
        self.time_avg_policy = np.zeros(self.pop_size)
        self.best_response = None
        self.policies_sum = np.zeros(self.pop_size)
        self.policy = np.zeros(self.pop_size)
        self.policy[self.current_strategies] = self.algo.policy

        self.ref_func = ref_func if ref_func is not None else lambda n: 50*np.log(n)
        self.refractory = self.ref_func(self.pop_size)
        self.ref_count = 0


    def update(self, cost_vector: np.ndarray, opponent):
        self.policy = np.zeros(self.pop_size)
        self.update_count += 1
        if len(self.current_strategies) != len(self.new_strategies):
            self.num_intervals += 1
            self.algo = re_init_no_regret(self.algo, self.current_strategies, self.lr, self.time_avg_policy,
                                          self.new_strategies, self.best_response)
            self.current_strategies = self.new_strategies[:]
            self.count = 1
            self.losses = np.zeros(self.pop_size)
            self.policy[self.new_strategies] = self.algo.policy
            if not self.whole_time_average:
                self.policies_sum = np.zeros(self.pop_size)
                self.time_avg_policy = np.zeros(self.pop_size)
                self.policies_sum[self.new_strategies] = self.algo.policy
                self.time_avg_policy[self.new_strategies] = self.algo.policy
            else:
                new_policies_sum = np.zeros(self.pop_size)
                new_policies_sum[:-1] = self.policies_sum[:]
                self.policies_sum = new_policies_sum
                self.time_avg_policy = self.policies_sum / self.update_count
        else:
            self.count += 1
            self.losses += cost_vector
            self.ref_count -= 1

            # Find best response and add if not already in leduc_pop
            if self.ref_count<=0:
                agg_agent = opponent.leduc_pop.agg_agents(opponent.time_avg_policy)
                self.exp, br_hagent = get_exploitability(agg_agent, skip_illegal=True)

                self.exps.append(self.exp)
                if not self.leduc_pop.hagent_in_pop(br_hagent):  # If this best response is not in leduc_pop
                    self.leduc_pop.add_agent(br_hagent)
                    self.pop_size += 1
                    self.new_strategies.append(self.current_strategies[-1]+1)
                    self.best_response = self.new_strategies[-1]
                self.ref_count = self.ref_func(self.pop_size)

            self.algo.update(cost_vector[self.current_strategies])
            self.policy[self.current_strategies] = self.algo.policy
            self.policies_sum += self.policy
            if self.whole_time_average:
                self.time_avg_policy = self.policies_sum / self.update_count
            else:
                self.time_avg_policy = self.policies_sum / self.count
            self.time_avg_policy /= np.sum(self.time_avg_policy)
            assert np.isclose(np.sum(self.time_avg_policy), 1.0)


if __name__ == "__main__":

    # Testing LeducPop and Metagame
    mgame = LeducMetagame(pop_size1=3, pop_size2=5, seed=0)

    r = np.random.RandomState(33)
    br_hagent = {}
    for infostate in mgame.env.infostates:
        br_hagent[infostate] = np.zeros(3)
        valid_actions = np.nonzero(mgame.env.illegal_action_dict[infostate])[0]
        a_idx = r.choice(valid_actions, 1)[0]
        br_hagent[infostate][a_idx] = 1.

    mgame.add_br_to_pop(br_hagent, player_index=0)

    print(mgame.metagame.shape)
    print(mgame.metagame)
