# Trading System for Economic AI Agents
# Implements spatial trading with offers and acceptance phases

from __future__ import annotations
import random
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
import config

if TYPE_CHECKING:
    from agent import Agent


@dataclass
class TradeOffer:
    """Represents a trade offer from one agent to another"""
    offerer: str  # Agent name making the offer
    target: str   # Agent name receiving the offer
    offer_red: int     # Red resources offered
    offer_green: int   # Green resources offered
    request_red: int   # Red resources requested
    request_green: int # Green resources requested
    step: int     # Step when offer was made
    
    def __str__(self):
        return f"{self.offerer}→{self.target}: Offer({self.offer_red}R,{self.offer_green}G) for ({self.request_red}R,{self.request_green}G)"


class Market:
    """Manages all trading activities between agents"""
    
    def __init__(self):
        self.offers: List[TradeOffer] = []
        self.completed_trades: List[TradeOffer] = []
        self.trade_log: List[str] = []
    
    def add_offer(self, offer: TradeOffer) -> bool:
        """Add a trade offer to the market"""
        self.offers.append(offer)
        self.trade_log.append(f"Step {offer.step}: {offer}")
        return True
    
    def get_offers_for_agent(self, agent_name: str) -> List[TradeOffer]:
        """Get all offers directed to a specific agent"""
        return [offer for offer in self.offers if offer.target == agent_name]
    
    def remove_offer(self, offer: TradeOffer):
        """Remove an offer from the market (when accepted or rejected)"""
        if offer in self.offers:
            self.offers.remove(offer)
    
    def execute_trade(self, offer: TradeOffer, agents_dict: Dict[str, 'Agent']) -> bool:
        """Execute a trade between two agents"""
        offerer = agents_dict.get(offer.offerer)
        target = agents_dict.get(offer.target)
        
        if not offerer or not target or not offerer.alive or not target.alive:
            return False
        
        # Check if both agents still have the required resources
        if (offerer.inventory['red'] >= offer.offer_red and 
            offerer.inventory['green'] >= offer.offer_green and
            target.inventory['red'] >= offer.request_red and
            target.inventory['green'] >= offer.request_green):
            
            # Execute the trade
            offerer.inventory['red'] -= offer.offer_red
            offerer.inventory['green'] -= offer.offer_green
            offerer.inventory['red'] += offer.request_red
            offerer.inventory['green'] += offer.request_green
            
            target.inventory['red'] -= offer.request_red
            target.inventory['green'] -= offer.request_green
            target.inventory['red'] += offer.offer_red
            target.inventory['green'] += offer.offer_green
            
            self.completed_trades.append(offer)
            self.trade_log.append(f"Step {offer.step}: TRADE EXECUTED - {offer}")
            return True
        
        return False
    
    def clear_offers(self):
        """Clear all pending offers (called at end of trading phase)"""
        self.offers.clear()
    
    def get_trade_summary(self) -> Dict:
        """Get summary statistics about trading activity"""
        return {
            "total_offers": len(self.trade_log),
            "completed_trades": len(self.completed_trades),
            "pending_offers": len(self.offers),
            "trade_log": self.trade_log.copy()
        }


def calculate_manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_agents_in_trade_range(agent: 'Agent', all_agents: List['Agent'], trade_range: int) -> List['Agent']:
    """Get all living agents within trade range of the given agent"""
    if trade_range <= 0:
        return []
    
    nearby_agents = []
    for other_agent in all_agents:
        if (other_agent.alive and 
            other_agent.name != agent.name and
            calculate_manhattan_distance(agent.position, other_agent.position) <= trade_range):
            nearby_agents.append(other_agent)
    
    return nearby_agents


def process_trading_phase(agents: List['Agent'], market: Market, step: int) -> List[str]:
    """
    Process the complete trading phase for all agents
    Returns list of trading events for logging
    """
    if config.TRADE_RANGE <= 0:
        return []
    
    trading_events = []
    agents_dict = {agent.name: agent for agent in agents}
    
    # Phase 1: All agents make offers simultaneously
    trading_events.append(f"=== TRADING PHASE STEP {step} ===")
    trading_events.append("Phase 1: Making offers...")
    
    for agent in agents:
        if agent.alive:
            # Get agents in trade range
            nearby_agents = get_agents_in_trade_range(agent, agents, config.TRADE_RANGE)
            
            if nearby_agents:
                # Agent decides whether to make a trade offer
                # For now, we'll implement basic logic - later this will use LLM
                offer = agent.consider_trade_offer(nearby_agents, step)
                if offer:
                    market.add_offer(offer)
                    trading_events.append(f"  {offer}")
    
    # Phase 2: Process acceptances in random order
    trading_events.append("Phase 2: Processing acceptances...")
    
    # Get all agents who have received offers
    agents_with_offers = []
    for agent in agents:
        if agent.alive:
            offers = market.get_offers_for_agent(agent.name)
            if offers:
                agents_with_offers.append((agent, offers))
    
    # Randomize order of acceptance processing
    random.shuffle(agents_with_offers)
    
    for agent, offers in agents_with_offers:
        # Agent decides which offer (if any) to accept
        accepted_offer = agent.consider_trade_acceptance(offers, step)
        
        if accepted_offer:
            # Execute the trade
            success = market.execute_trade(accepted_offer, agents_dict)
            if success:
                trading_events.append(f"  ACCEPTED: {accepted_offer}")
                # Remove the accepted offer and any conflicting offers
                market.remove_offer(accepted_offer)
                
                # Remove other offers to/from these agents to prevent conflicts
                conflicting_offers = [
                    offer for offer in market.offers 
                    if (offer.offerer == accepted_offer.offerer or 
                        offer.target == accepted_offer.offerer or
                        offer.offerer == accepted_offer.target or 
                        offer.target == accepted_offer.target)
                ]
                for conflict in conflicting_offers:
                    market.remove_offer(conflict)
                    trading_events.append(f"  CANCELLED (conflict): {conflict}")
            else:
                trading_events.append(f"  FAILED: {accepted_offer} (insufficient resources)")
                market.remove_offer(accepted_offer)
        else:
            # Reject all offers
            for offer in offers:
                trading_events.append(f"  REJECTED: {offer}")
                market.remove_offer(offer)
    
    # Clear any remaining offers
    market.clear_offers()
    trading_events.append("Phase 2 complete - all offers cleared")
    
    return trading_events
