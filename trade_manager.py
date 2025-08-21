class TradeManager:
    def __init__(self):
        self.offers = []
        self.next_offer_id = 1

    def make_offer(self, from_agent, give: dict, want: dict):
        offer = {
            'id': self.next_offer_id,
            'from': from_agent.name,
            'give': give,
            'want': want,
            'status': 'open'
        }
        self.offers.append(offer)
        self.next_offer_id += 1
        return offer

    def get_open_offers(self, excluding_agent=None):
        return [
            o for o in self.offers
            if o['status'] == 'open' and o['from'] != excluding_agent
        ]

    def accept_offer(self, offer_id, to_agent):
        offer = next((o for o in self.offers if o['id'] == offer_id and o['status'] == 'open'), None)
        if not offer:
            return "Offer not found or already accepted."

        offer['status'] = 'accepted'
        return offer

    def list_offers(self):
        return self.offers
