"""
A not very elegant but functional implementation of Leduc Poker based on open_spiel leduc poker
"""

import numpy as np
import random
import itertools
import copy

LEDUC_ACTIONS = ['f', 'r', 'c']  # fold, raise, call
LEDUC_CARDS = ['J1', 'Q1', 'K1', 'J2', 'Q2', 'K2']
LEDUC_FIRST_RAISE_AMOUNT = 2
LEDUC_SECOND_RAISE_AMOUNT = 4



class LeducPoker:
    '''
    Follow Figure 1 in poker.cs.ualberta.ca/publications/UAI05.pdf

    The game tree:
    1. There are 6 cards of which 2 are chosen for players 0 and 1 respectively
        That is a total of V_2(6)=6!/(6-2)!=6*5=30 subtrees.
    2. Let's take 1 subtree. In this subtree there can be 4 paths in which we do not draw a common card:
        These correspond to folding in first round rf, rrf, crf, crrf.
    3. The other 5 paths lead to a chance node where we take a card rc., rrc., cc., crc., crrc.
    4. There are 4 different subtrees from these five leafs corresponding to picking a card from the 4 remaining
        on the deck. This is a total of 5*4
    5. Then, in each of the 5*24 leafs there are a total of 9 possible paths corresponding to second rounds like
        .rf, .rrf, .rrc, .rc, .crf, .crrf, .crrc, .crc, .cc
    6. This means there is a total of 5*4*9=180 different endings
    7. Adding the 4 endings and considering this is just 1 subtree of 30 leads to a total of 184*30=5520

    '''

    def __init__(self):
        self.done = False
        self.current_player = 0
        self.current_cards = None
        self.common_card = None
        self.deck = None  # Cards remaining in the deck
        self.history = None
        self.stakes = None
        self.ante = None
        self.nb_infostates = 936  # If suits not relevant then 288
        self.first_raise_amount = LEDUC_FIRST_RAISE_AMOUNT
        self.second_raise_amount = LEDUC_SECOND_RAISE_AMOUNT

        self.infostates, _, self.illegal_action_dict = LeducPoker.get_infostates()
        assert len(self.infostates) == 936  # 288
        self.actions = LEDUC_ACTIONS
        self.cards = LEDUC_CARDS

    def reset(self, set_cards=None):
        self.current_player = 0
        self.done = False
        self.current_cards = list(random.sample(self.cards, 2) if set_cards is None else set_cards)
        self.deck = [card for card in LEDUC_CARDS if card not in self.current_cards]
        self.history = ''
        self.stakes = 1
        self.ante = [1, 1]
        infostate = self.current_cards[self.current_player] + self.history
        return [self.infostate2vector(infostate), infostate]

    def step(self, a):

        # Then apply player action
        a = self.actions[a]
        self.history, self.stakes, self.ante = LeducPoker.apply_action(self.history, self.stakes, self.ante[:], a, self.current_player)
        self.current_player = 1 - self.current_player  # Next player

        if self.is_terminal(self.history):
            self.done = True
            r = LeducPoker.get_payoff(self.history, self.current_cards+[self.common_card], self.ante, self.current_player)
            next_obs = [0., 0.]
        else:
            # Solve chance node if necessary
            if len(self.history) and self.history[-1]=='.' and self.history[-2:]!='..':
                self.common_card = self.deck[np.random.choice(4, 1)[0]]
                self.history += '.'

            round = self.history.count('..') + 1
            # Note this infostate is for next player since he needs the observation with his cards to choose an action
            infostate = self.current_cards[self.current_player] + self.history if round == 1 \
                else self.current_cards[self.current_player] + self.common_card + self.history
            next_obs = [self.infostate2vector(infostate), infostate]
            r = (0., 0.)

        return next_obs, r, self.done, None

    def infostate2vector(self, infostate):
        assert infostate in self.infostates, infostate + " not in infostates"
        idx = self.infostates.index(infostate)
        vec = np.zeros((len(self.infostates),), dtype=np.float32)
        vec[idx] = 1.
        return vec


    @staticmethod
    def action_is_illegal(a, round_history, stakes, ante, current_player):
        return (a == 'r' and round_history.count('r') >= 2) or (a == 'f' and stakes <= ante[current_player])

    @staticmethod
    def get_current_player(history):
        """ Returns the player to play next turn """
        h = history.replace('..', '') if '..' in history else history.replace('.', '')
        return len(h) % 2

    @staticmethod
    def rank(cards, round):
        """ Returns the number of the winning player (-1 if draw). Lowest rank wins"""
        # We first convert suits to values e.g. K1->K, J2->J
        tmp = copy.deepcopy(cards)
        cards = ['X']*3
        cards[0] = tmp[0][0]
        cards[1] = tmp[1][0]
        cards[2] = tmp[2][0]

        # If round 1 only use 1 card
        if round==1:
            ranks = {'K': 1, 'Q': 2, 'J': 3}
            rank1 = ranks[cards[0]]  # Player 1
            rank2 = ranks[cards[1]]  # Player 2
        else:
            ranks = {
                'KK': 1, 'QQ': 2, 'JJ': 3,
                'KQ': 4, 'QK': 4, 'KJ': 5,
                'JK': 5, 'QJ': 6, 'JQ': 6}
            rank1 = ranks[cards[0]+cards[2]]  # Player 1
            rank2 = ranks[cards[1]+cards[2]]  # Player 2
        if rank1 < rank2:   # Player 1 winner
            return 0
        elif rank1==rank2:  # Draw
            return -1
        else:               # Player 2 winner
            return 1

    @staticmethod
    def get_payoff(history, cards, ante, current_player):
        """ Returns payoff for each player. Assumes is_terminal(history) is True.
            cards is a list [p1_card, p2_card] if round or [p1_card, p2_card, common_card] if round 2
            Current player refers to the player who would play now if game had not ended
            - If current player wins, payoff is the other player's ante
            - If current player loss, payoff is the other player's negative his ante
            - If draw, both payoffs are zero
        """
        # Fold
        if history[-1]=='f':  # Current player wins
            if current_player==0:
                return ante[1], -ante[1]
            else:
                return -ante[0], ante[0]
        # Showdown!
        else:
            round = history.count('..') + 1
            assert round==1 or round==2, "Wrong number of rounds"
            #  First and second cards belong to player 1 and 2 respectively. Third card is common
            winner = LeducPoker.rank(cards, round)
            if winner==-1:  # Draw
                assert ante[0]-ante[1]==0
                return 0, 0
            elif winner==current_player:  # Current player wins
                if current_player==0:
                    return ante[1], -ante[1]
                else:
                    return -ante[0], ante[0]
            else:                         # Current player loses
                if current_player==0:
                    return -ante[0], ante[0]
                else:
                    return ante[1], -ante[1]

    @staticmethod
    def is_terminal(history: str) -> bool:
        """ Game is only finished if one player folds or second round is finished """
        if len(history)==0:
            return False
        return history[-1]=='f' or (LeducPoker.ready_for_next_round(history) and history.count('..')>0)

    @staticmethod
    def ready_for_next_round(history):
        """ Check if current round finished. True if any:
                - Two calls in a row
                - Started raising and there was one call (see tree)
                - Started calling and there was two calls (see tree)
        """
        round_history = history.split('..')[-1]
        if len(round_history)==0:
            return False
        return ((round_history.count('r')==0 and round_history.count('c')==2) or
                (round_history[0]=='r' and round_history.count('c')==1 ) or
                (round_history[0]=='c' and round_history.count('c')==2 )
                )

    @staticmethod
    def apply_action(history, stakes, ante, a, current_player):
        """
          Applies a player action (chance nodes already solved) to the current history, raises the stakes and ante
          ante: what the player has contributed to the pot
          stakes: current level of the bet
        """

        # 0. Current round
        round = history.count('..') + 1
        round_history = history.split('..')[-1]

        # 1. Map illegal actions to call
        if a=='r' and round_history.count('r')>=2:  # Max of two raises per round
            a = 'c'
        elif a=='f' and stakes<=ante[current_player]:  # Player can only fold if his ante is less than stakes
            a = 'c'

        # 2. Fold
        if a=='f':
            history += a
            return history, stakes, ante
        # 3. Call
        elif a=='c':
            # Current player puts in an amount of money equal to the current level (stakes) minus what they have
            # contributed to level their contribution off.
            amount = stakes - ante[current_player]
            assert amount>=0, "Error ante > stakes when calling ('c') "
            ante[current_player] += amount
            history += a
            if LeducPoker.is_terminal(history):
                return history, stakes, ante
            elif LeducPoker.ready_for_next_round(history):
                history += '.'  # We add '.' to indicate round 1 was finished and we need to solve a chance node
                return history, stakes, ante
            else:
                return history, stakes, ante
        # 4. Raise
        elif a=='r':
            # First match current stakes
            call_amount = stakes - ante[current_player]
            assert call_amount>=0, "Error ante > stakes when raising ('r') "
            ante[current_player] += call_amount
            # Now raise the stakes
            raise_amount = LEDUC_FIRST_RAISE_AMOUNT if round==1 else LEDUC_SECOND_RAISE_AMOUNT
            stakes += raise_amount
            ante[current_player] += raise_amount
            history += a
            return history, stakes, ante

    @staticmethod
    def traverse_game_tree(history, stakes, ante, cards, infostates, end_games, illegal_action_dict):
        """ Traverses the the whole game tree recursively for a given set of three cards.
            Records all distinct infostates and all finished game histories (note we use python list mutability)
        """
        if LeducPoker.is_terminal(history):
            # Add an end game (history of game that has finished and its payoff)
            round, current_player = history.count('.') + 1, LeducPoker.get_current_player(history)
            key = cards[0]+cards[1]+history if round==1 else cards[0]+cards[1]+cards[2]+history
            new_key, counter = key, 1
            if new_key in end_games:
                assert end_games[new_key] == LeducPoker.get_payoff(history, cards, ante, current_player), \
                    'ERROR: Same finished histories must have same payoff'
                return
            end_games[new_key] = LeducPoker.get_payoff(history, cards, ante, current_player)
            return

        # Chance node
        if len(history) and history[-1]=='.' and history[-2:]!='..':
            deck = [card for card in LEDUC_CARDS if card not in cards]
            for common_card in deck:
                new_history = history + '.'
                new_cards = cards + [common_card]
                LeducPoker.traverse_game_tree(new_history, stakes, ante[:], new_cards[:], infostates, end_games, illegal_action_dict)

        # Player node
        else:
            for a_idx, a in enumerate(LEDUC_ACTIONS):
                current_player = LeducPoker.get_current_player(history)
                round = history.count('..') + 1
                round_history = history.split('..')[-1]
                key = cards[current_player] + history if round==1 else cards[current_player] + cards[2] + history

                if key not in illegal_action_dict:
                    illegal_action_dict[key] = np.array([1, 1, 1])

                # Skip illegal actions
                if LeducPoker.action_is_illegal(a, round_history, stakes, ante[:], current_player):
                    illegal_action_dict[key][a_idx] = 0
                    continue

                # Add current infostate key
                if key not in infostates:
                    infostates.append(key)

                # Apply action
                new_history, new_stakes, new_ante = LeducPoker.apply_action(history, stakes, ante[:], a, current_player)  # Note we use ante[:] to avoid list mutability

                # Continue to subtree
                LeducPoker.traverse_game_tree(new_history, new_stakes, new_ante[:], cards, infostates, end_games, illegal_action_dict)

    @staticmethod
    def get_infostates():
        """
            Computes all infostates and end histories by traversing the whole game tree for each possible drawing of the
             first three cards (30*4=120).
            We user itertools.permutations(LEDUC_CARDS, 2) to choose all 2 from a deck of 6 cards
            considering order important and no repetition (this is actually Variations without repetition)
        """
        infostates = []
        end_games = {}
        illegal_action_dict = {}
        for cards in itertools.permutations(LEDUC_CARDS, 2):
            cards = list(cards)
            history = ""
            stakes = 1
            ante = [1, 1]
            LeducPoker.traverse_game_tree(history, stakes, ante, cards, infostates, end_games, illegal_action_dict)
        return infostates, end_games, illegal_action_dict


def calc_ev(p1_strat, p2_strat, cards, history, stakes, ante, current_player, skip_illegal=False):
    """ Computes the exact expected payoff of two strategies for player 1 and player 2 respectively
        - In each node, we multiply the probability of picking each child subtree times the expected payoff
        of the subtree recursively.
        - We do not correct for the first two cards. This can be achieved by calling this function for each Variation
        without repetition of six cards taken two by two V_2(6)=6*5=30 adding all expected payoffs
        and multiplying it times 1/30.
        Example:
        for cards in itertools.permutations(LEDUC_CARDS, 2):
            cards, history, stakes, ante, currentplayer = list(cards), "", 1, [1, 1], 0
            v1, v2 = calc_ev(p1_strat, p2_strat, cards, history, stakes, ante, current_player)
            value1 += v1/30.
            value2 += v2/30.
    """

    # Return payoff is terminal node
    if LeducPoker.is_terminal(history):
        return LeducPoker.get_payoff(history, cards, ante, current_player)

    # Chance node
    if len(history) and history[-1]=='.' and history[-2:]!='..':
        deck = [card for card in LEDUC_CARDS if card not in cards]
        chances = 1./4 * np.ones((4,))
        ev0, ev1 = [], []
        for common_card in deck:  # Each possible common card draw
            new_history = history + '.'
            new_cards = cards + [common_card]
            ev = calc_ev(p1_strat, p2_strat, new_cards[:], new_history, stakes, ante[:], current_player, skip_illegal=skip_illegal)
            ev0.append(ev[0])
            ev1.append(ev[1])
        return np.dot(chances, ev0), np.dot(chances, ev1)

    # Player node
    else:
        # Compute infostate key
        round = history.count('.') + 1
        round_history = history.split('..')[-1]
        key = cards[current_player] + history if round==1 else cards[current_player] + cards[2] + history
        strat = p1_strat[key] if current_player==0 else p2_strat[key]  # These are the probs. of choosing the three subtrees

        # Apply action
        next_player = 1-current_player
        ev0, ev1 = [], []
        for a in LEDUC_ACTIONS:
            # Skip illegal actions
            if skip_illegal and LeducPoker.action_is_illegal(a, round_history, stakes, ante[:], current_player):
                ev0.append(0.)
                ev1.append(0.)
                continue

            # Apply action and compute payoff of subtree
            new_history, new_stakes, new_ante = LeducPoker.apply_action(history, stakes, ante[:], a, current_player)
            ev = calc_ev(p1_strat, p2_strat, cards, new_history, new_stakes, new_ante[:], next_player, skip_illegal=skip_illegal)
            ev0.append(ev[0])
            ev1.append(ev[1])
        return np.dot(strat, ev0), np.dot(strat, ev1)


def calc_best_response(agg_hagent, br_strat_map, br_player, cards, history, stakes, ante, current_player, prob, skip_illegal=False):
    """
    After first chance node, cards already dealt
    """
    # Return payoff is terminal node
    if LeducPoker.is_terminal(history):
        payoffs = LeducPoker.get_payoff(history, cards, ante, current_player)
        return payoffs[br_player]

    # Chance node
    if len(history) and history[-1]=='.' and history[-2:]!='..':
        deck = [card for card in LEDUC_CARDS if card not in cards]
        chances = 1./4 * np.ones((4,))
        chance_values = []
        for idx, common_card in enumerate(deck):
            new_history = history + '.'
            new_cards = cards + [common_card]
            new_prob = prob * chances[idx]
            chance_value = calc_best_response(agg_hagent, br_strat_map, br_player, new_cards, new_history, stakes,
                                              ante[:], current_player, new_prob, skip_illegal=skip_illegal)
            chance_values.append(chance_value)
        return np.dot(chances, chance_values)

    else:
        # Compute infostate key
        round = history.count('..') + 1
        round_history = history.split('..')[-1]
        key = cards[current_player] + history if round==1 else cards[current_player] + cards[2] + history

        next_player = 1 - current_player
        if current_player == br_player:
            vals = []
            for a in LEDUC_ACTIONS:

                # Skip illegal actions
                if skip_illegal and LeducPoker.action_is_illegal(a, round_history, stakes, ante[:], current_player):
                    vals.append(-1e8)  # Make this subtree value very bad (won't ever be in a BR)
                    continue

                new_history, new_stakes, new_ante = LeducPoker.apply_action(history, stakes, ante[:], a, current_player)
                val = calc_best_response(agg_hagent, br_strat_map, br_player, cards, new_history, new_stakes, new_ante[:], next_player, prob, skip_illegal=skip_illegal)
                vals.append(val)

            best_response_value = max(vals)
            if key not in br_strat_map:
                br_strat_map[key] = np.array([0., 0., 0.])
            br_strat_map[key] = br_strat_map[key] + prob * np.array(vals, dtype=np.float64)
            return best_response_value
        else:
            strategy = agg_hagent[key]
            action_values = []
            for idx, a in enumerate(LEDUC_ACTIONS):

                # Skip illegal actions
                if skip_illegal and LeducPoker.action_is_illegal(a, round_history, stakes, ante[:], current_player):
                    action_values.append(-1e8)  # Make this subtree value very bad (won't ever be in a BR)
                    continue

                new_history, new_stakes, new_ante = LeducPoker.apply_action(history, stakes, ante[:], a, current_player)
                new_prob = prob*strategy[idx]
                action_value = calc_best_response(agg_hagent, br_strat_map, br_player, cards, new_history, new_stakes, new_ante[:], next_player, new_prob, skip_illegal=skip_illegal)
                action_values.append(action_value)
            return np.dot(strategy, action_values)


def get_exploitability(agg_hagent, skip_illegal=False):
    exploitability = 0.

    br_strat_map = {}
    for cards in itertools.permutations(LEDUC_CARDS, 2):
        cards = list(cards)
        history, stakes, ante, current_player = "", 1, [1, 1], 0
        calc_best_response(agg_hagent, br_strat_map, 0, cards, history, stakes, ante[:], current_player, 1.0, skip_illegal=skip_illegal)
        history, stakes, ante, current_player = "", 1, [1, 1], 0
        calc_best_response(agg_hagent, br_strat_map, 1, cards, history, stakes, ante[:], current_player, 1.0, skip_illegal=skip_illegal)

    for k,v in br_strat_map.items():
        v[:] = np.where(v == np.max(v), 1, 0)
        if np.sum(v)>1.:
            idxs = np.nonzero(v)
            v[:] = np.zeros_like(v)
            v[np.random.choice(idxs[0])] = 1.

    for cards in itertools.permutations(LEDUC_CARDS, 2):
        cards = list(cards)
        history, stakes, ante, current_player = "", 1, [1, 1], 0
        tmp1, ev_2 = calc_ev(agg_hagent, br_strat_map, cards[:], history, stakes, ante[:], current_player, skip_illegal=skip_illegal)
        ev_1, tmp2 = calc_ev(br_strat_map, agg_hagent, cards[:], history, stakes, ante[:], current_player, skip_illegal=skip_illegal)
        exploitability += (1 / 30) * (ev_2 + ev_1)
    return exploitability/2., br_strat_map


def leduc_reset():
    history, stakes, ante, current_player = "", 1, [1, 1], 0
    return history, stakes, ante, current_player



if __name__=="__main__":

    skip = False

    leduc_game = LeducPoker()
    infostates, end_games, illegal_action_dict = leduc_game.get_infostates()
    print(len(infostates))
    print(len(end_games))
    assert len(infostates) == 936  # 288 if suits don't matter
    assert len(end_games) == 5520

    p1_strat = {}
    p2_strat = {}

    for infostate in infostates:
        p = np.random.rand(2, 3)
        p1_strat[infostate] = p[0, :]/np.sum(p[0, :])
        p2_strat[infostate] = p[1, :]/np.sum(p[1, :])

    value1, value2 = 0., 0.
    for cards in itertools.permutations(LEDUC_CARDS, 2):
        cards = list(cards)
        history = ""
        stakes = 1
        ante = [1, 1]
        current_player = 0
        v1, v2 = calc_ev(p1_strat, p2_strat, cards, history, stakes, ante, current_player, skip_illegal=skip)
        value1 += v1/30.
        value2 += v2/30.
    print(value1, value2)

    import time
    start = time.time()
    exploitability, br_strat_map = get_exploitability(p1_strat, skip_illegal=skip)
    print(time.time()-start)
    print(exploitability)






