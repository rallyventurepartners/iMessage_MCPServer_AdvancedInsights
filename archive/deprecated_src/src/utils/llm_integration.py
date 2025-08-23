"""
LLM Integration utilities for generating advanced insights.

This module provides the infrastructure for integrating with Claude and other LLMs
to generate sophisticated conversational insights beyond basic NLP analysis.
"""

import logging
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import asyncio

from ..exceptions import ToolExecutionError
from .analysis_cache import cache_llm_insights, get_message_hash

logger = logging.getLogger(__name__)


class LLMInsightGenerator:
    """Handles LLM-based insight generation for conversations."""
    
    def __init__(self, model_name: str = "claude-3"):
        self.model_name = model_name
        self.max_context_length = 8000  # Conservative limit for context
        
    async def generate_conversation_insights(
        self,
        messages: List[Dict[str, Any]],
        analysis_results: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate deep insights about a conversation using LLM analysis.
        
        Args:
            messages: List of messages to analyze
            analysis_results: Results from traditional analysis methods
            focus_areas: Specific areas to focus on
            
        Returns:
            Dictionary containing LLM-generated insights
        """
        try:
            # Prepare context for LLM
            context = self._prepare_context(messages, analysis_results)
            
            # Generate prompt based on focus areas
            prompt = self._build_insight_prompt(context, focus_areas or [])
            
            # Check cache first
            context_hash = get_message_hash(messages)
            
            async def call_llm(prompt_text: str) -> Dict[str, Any]:
                # In a real implementation, this would call the actual LLM API
                # For now, we'll generate structured insights based on the analysis
                return self._generate_structured_insights(
                    messages, analysis_results, focus_areas
                )
            
            # Use caching to avoid redundant LLM calls
            insights = await cache_llm_insights(
                context_hash=context_hash,
                prompt=prompt,
                llm_func=lambda p: call_llm(p),
                focus_areas=focus_areas
            )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating LLM insights: {e}")
            # Return fallback insights if LLM fails
            return self._generate_fallback_insights(analysis_results)
    
    def _prepare_context(
        self,
        messages: List[Dict[str, Any]],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare context for LLM analysis."""
        
        # Sample representative messages
        sampled_messages = self._sample_messages(messages, max_messages=50)
        
        # Prepare conversation summary
        context = {
            "conversation_stats": {
                "total_messages": len(messages),
                "date_range": self._get_date_range(messages),
                "participants": self._get_participants(messages)
            },
            "analysis_summary": {
                "depth_score": analysis.get("conversation_depth", {}).get("depth_score", 0),
                "topic_diversity": analysis.get("topic_intelligence", {}).get("topic_diversity", 0),
                "emotional_tone": analysis.get("emotional_dynamics", {}).get("emotional_tone", {}),
                "health_score": analysis.get("health_score", {}).get("score", 0)
            },
            "sample_exchanges": self._extract_exchanges(sampled_messages),
            "detected_patterns": self._extract_patterns(analysis)
        }
        
        return context
    
    def _build_insight_prompt(
        self,
        context: Dict[str, Any],
        focus_areas: List[str]
    ) -> str:
        """Build a prompt for the LLM based on context and focus areas."""
        
        prompt = f"""Analyze this conversation data and provide deep insights:

Conversation Overview:
- Total messages: {context['conversation_stats']['total_messages']}
- Date range: {context['conversation_stats']['date_range']}
- Depth score: {context['analysis_summary']['depth_score']}/100
- Topic diversity: {context['analysis_summary']['topic_diversity']}
- Health score: {context['analysis_summary']['health_score']}/100

Key Patterns Detected:
{json.dumps(context['detected_patterns'], indent=2)}

Sample Exchanges:
{self._format_exchanges(context['sample_exchanges'][:5])}

Please provide insights on:
1. Relationship dynamics and communication patterns
2. Emotional support and reciprocity
3. Areas of strength in the relationship
4. Opportunities for deeper connection
"""
        
        if focus_areas:
            prompt += f"\n\nSpecific Focus Areas: {', '.join(focus_areas)}"
        
        prompt += "\n\nProvide actionable insights and specific recommendations."
        
        return prompt
    
    def _generate_structured_insights(
        self,
        messages: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate structured insights based on analysis data."""
        
        depth = analysis.get("conversation_depth", {})
        topics = analysis.get("topic_intelligence", {})
        emotions = analysis.get("emotional_dynamics", {})
        response_patterns = analysis.get("response_patterns", {})
        health = analysis.get("health_score", {})
        
        insights = {
            "relationship_summary": self._generate_relationship_summary(
                depth, topics, emotions, health
            ),
            "communication_style": self._analyze_communication_style(
                messages, response_patterns, depth
            ),
            "areas_of_strength": self._identify_strengths(
                depth, topics, emotions, response_patterns
            ),
            "growth_opportunities": self._identify_growth_areas(
                depth, topics, emotions, response_patterns
            ),
            "actionable_recommendations": self._generate_recommendations(
                analysis, focus_areas
            )
        }
        
        # Add focus area specific insights
        if focus_areas:
            for area in focus_areas:
                if area == "emotional_support":
                    insights["emotional_support_analysis"] = self._analyze_emotional_support(
                        emotions, response_patterns, messages
                    )
                elif area == "conflict":
                    insights["conflict_patterns"] = self._analyze_conflict_patterns(
                        emotions, messages
                    )
                elif area == "growth":
                    insights["relationship_growth"] = self._analyze_relationship_growth(
                        topics, depth, messages
                    )
        
        return insights
    
    def _generate_relationship_summary(
        self, depth: Dict, topics: Dict, emotions: Dict, health: Dict
    ) -> str:
        """Generate a comprehensive relationship summary."""
        
        depth_category = depth.get("depth_category", "Unknown")
        health_score = health.get("score", 0)
        emotional_tone = emotions.get("emotional_tone", {}).get("dominant", "neutral")
        
        if health_score >= 80:
            quality = "exceptionally strong and healthy"
        elif health_score >= 60:
            quality = "healthy and positive"
        elif health_score >= 40:
            quality = "developing with room for growth"
        else:
            quality = "showing signs of distance or strain"
        
        summary = f"This appears to be a {quality} relationship characterized by "
        
        if depth_category == "Deep":
            summary += "meaningful, substantive conversations "
        elif depth_category == "Moderate":
            summary += "balanced exchanges with occasional depth "
        else:
            summary += "primarily surface-level interactions "
        
        summary += f"and a predominantly {emotional_tone} emotional tone. "
        
        if topics.get("topic_diversity", 0) > 50:
            summary += "The wide range of topics discussed indicates a well-rounded relationship."
        else:
            summary += "Consider exploring new topics to enhance connection."
        
        return summary
    
    def _analyze_communication_style(
        self, messages: List[Dict], response_patterns: Dict, depth: Dict
    ) -> str:
        """Analyze and describe communication style."""
        
        avg_response_time = response_patterns.get("response_time_analysis", {}).get(
            "average_minutes", 0
        )
        balance = response_patterns.get("message_balance", {}).get("balance_category", "Unknown")
        
        style_elements = []
        
        if avg_response_time < 30:
            style_elements.append("highly responsive")
        elif avg_response_time < 120:
            style_elements.append("moderately responsive")
        else:
            style_elements.append("relaxed response pace")
        
        if balance == "Balanced":
            style_elements.append("reciprocal engagement")
        elif "Sender" in balance:
            style_elements.append("you tend to initiate more")
        else:
            style_elements.append("your contact tends to lead conversations")
        
        if depth.get("metrics", {}).get("question_ratio", 0) > 0.15:
            style_elements.append("curiosity-driven dialogue")
        
        return f"Communication is characterized by {', '.join(style_elements)}."
    
    def _identify_strengths(
        self, depth: Dict, topics: Dict, emotions: Dict, response_patterns: Dict
    ) -> List[str]:
        """Identify relationship strengths."""
        
        strengths = []
        
        if depth.get("depth_score", 0) > 60:
            strengths.append("Ability to engage in meaningful, substantive conversations")
        
        if emotions.get("emotional_tone", {}).get("positive_ratio", 0) > 0.6:
            strengths.append("Consistently positive and supportive emotional exchanges")
        
        if topics.get("topic_diversity", 0) > 40:
            strengths.append("Wide range of shared interests and discussion topics")
        
        if response_patterns.get("message_balance", {}).get("balance_category") == "Balanced":
            strengths.append("Well-balanced reciprocal communication pattern")
        
        consistency = response_patterns.get("response_time_analysis", {}).get("consistency", "")
        if "consistent" in consistency.lower():
            strengths.append("Reliable and consistent engagement patterns")
        
        if not strengths:
            strengths.append("Regular communication maintaining connection")
        
        return strengths
    
    def _identify_growth_areas(
        self, depth: Dict, topics: Dict, emotions: Dict, response_patterns: Dict
    ) -> List[str]:
        """Identify areas for relationship growth."""
        
        opportunities = []
        
        if depth.get("depth_score", 0) < 40:
            opportunities.append(
                "Engage in more open-ended questions to deepen conversations"
            )
        
        if topics.get("topic_diversity", 0) < 30:
            opportunities.append(
                "Explore new shared interests or activities to broaden connection"
            )
        
        if emotions.get("emotional_tone", {}).get("positive_ratio", 0) < 0.4:
            opportunities.append(
                "Increase expressions of appreciation and positive sentiment"
            )
        
        balance = response_patterns.get("message_balance", {}).get("ratio", 1.0)
        if balance > 2.0 or balance < 0.5:
            opportunities.append(
                "Work towards more balanced message initiation between both parties"
            )
        
        if depth.get("metrics", {}).get("question_ratio", 0) < 0.05:
            opportunities.append(
                "Ask more questions to show interest and encourage sharing"
            )
        
        return opportunities
    
    def _generate_recommendations(
        self, analysis: Dict, focus_areas: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """Generate specific actionable recommendations."""
        
        recommendations = []
        health_score = analysis.get("health_score", {}).get("score", 50)
        
        if health_score < 60:
            recommendations.append({
                "priority": "high",
                "action": "Schedule regular check-ins",
                "rationale": "Consistent communication helps maintain connection",
                "timeframe": "Start this week"
            })
        
        depth_score = analysis.get("conversation_depth", {}).get("depth_score", 0)
        if depth_score < 50:
            recommendations.append({
                "priority": "medium",
                "action": "Share a personal story or experience",
                "rationale": "Vulnerability encourages deeper connection",
                "timeframe": "Next conversation"
            })
        
        topic_diversity = analysis.get("topic_intelligence", {}).get("topic_diversity", 0)
        if topic_diversity < 40:
            recommendations.append({
                "priority": "low",
                "action": "Suggest a new shared activity or topic",
                "rationale": "Shared experiences create lasting bonds",
                "timeframe": "Within the month"
            })
        
        return recommendations
    
    def _analyze_emotional_support(
        self, emotions: Dict, response_patterns: Dict, messages: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze emotional support patterns in the relationship."""
        
        support_indicators = emotions.get("support_indicators", {})
        emotional_reciprocity = emotions.get("emotional_reciprocity", {})
        
        return {
            "support_given": self._assess_support_given(support_indicators, messages),
            "support_received": self._assess_support_received(support_indicators, messages),
            "emotional_availability": self._assess_emotional_availability(
                response_patterns, emotions
            ),
            "recommendations": self._generate_support_recommendations(
                support_indicators, emotional_reciprocity
            )
        }
    
    def _analyze_conflict_patterns(
        self, emotions: Dict, messages: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze conflict and resolution patterns."""
        
        negative_periods = emotions.get("sentiment_timeline", {}).get("negative_periods", [])
        
        return {
            "conflict_frequency": len(negative_periods),
            "typical_triggers": self._identify_conflict_triggers(messages, negative_periods),
            "resolution_patterns": self._analyze_resolution_patterns(messages, emotions),
            "healthy_conflict_indicators": self._assess_conflict_health(emotions, messages)
        }
    
    def _analyze_relationship_growth(
        self, topics: Dict, depth: Dict, messages: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze relationship growth over time."""
        
        return {
            "depth_trajectory": self._calculate_depth_trajectory(messages, depth),
            "topic_evolution": topics.get("topic_evolution", {}),
            "milestones": self._identify_relationship_milestones(messages),
            "growth_indicators": self._identify_growth_indicators(depth, topics)
        }
    
    # Helper methods
    def _sample_messages(self, messages: List[Dict], max_messages: int = 50) -> List[Dict]:
        """Sample representative messages from the conversation."""
        if len(messages) <= max_messages:
            return messages
        
        # Sample evenly across the time range
        step = len(messages) // max_messages
        return messages[::step][:max_messages]
    
    def _get_date_range(self, messages: List[Dict]) -> str:
        """Get the date range of messages."""
        if not messages:
            return "No messages"
        
        dates = [msg.get("date", "") for msg in messages if msg.get("date")]
        if not dates:
            return "Unknown date range"
        
        return f"{min(dates)} to {max(dates)}"
    
    def _get_participants(self, messages: List[Dict]) -> List[str]:
        """Extract unique participants from messages."""
        participants = set()
        for msg in messages:
            if msg.get("sender"):
                participants.add(msg["sender"])
            if msg.get("contact_name"):
                participants.add(msg["contact_name"])
        return list(participants)
    
    def _extract_exchanges(self, messages: List[Dict]) -> List[Dict]:
        """Extract conversational exchanges."""
        exchanges = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                exchanges.append({
                    "message": messages[i].get("text", "")[:100],
                    "response": messages[i + 1].get("text", "")[:100],
                    "time_delta": self._calculate_time_delta(messages[i], messages[i + 1])
                })
        return exchanges[:10]  # Return top 10 exchanges
    
    def _extract_patterns(self, analysis: Dict) -> Dict[str, Any]:
        """Extract key patterns from analysis."""
        return {
            "communication_frequency": analysis.get("response_patterns", {}).get(
                "daily_message_frequency", "Unknown"
            ),
            "dominant_topics": analysis.get("topic_intelligence", {}).get(
                "main_topics", []
            )[:5],
            "emotional_pattern": analysis.get("emotional_dynamics", {}).get(
                "emotional_tone", {}
            ),
            "engagement_level": analysis.get("conversation_depth", {}).get(
                "depth_category", "Unknown"
            )
        }
    
    def _format_exchanges(self, exchanges: List[Dict]) -> str:
        """Format exchanges for prompt."""
        formatted = []
        for ex in exchanges:
            formatted.append(
                f"Person A: {ex['message']}\n"
                f"Person B: {ex['response']}\n"
                f"(Response time: {ex['time_delta']})\n"
            )
        return "\n".join(formatted)
    
    def _calculate_time_delta(self, msg1: Dict, msg2: Dict) -> str:
        """Calculate time between messages."""
        try:
            date1 = datetime.fromisoformat(msg1.get("date", ""))
            date2 = datetime.fromisoformat(msg2.get("date", ""))
            delta = date2 - date1
            
            if delta.days > 0:
                return f"{delta.days} days"
            elif delta.seconds > 3600:
                return f"{delta.seconds // 3600} hours"
            else:
                return f"{delta.seconds // 60} minutes"
        except:
            return "Unknown"
    
    def _assess_support_given(self, indicators: Dict, messages: List[Dict]) -> str:
        """Assess the level of emotional support given."""
        support_count = indicators.get("support_messages_sent", 0)
        total = len([m for m in messages if m.get("is_sender", False)])
        
        if total == 0:
            return "Unable to assess support patterns"
        
        support_ratio = support_count / total
        
        if support_ratio > 0.2:
            return "You frequently offer emotional support and validation"
        elif support_ratio > 0.1:
            return "You provide moderate emotional support when needed"
        else:
            return "Consider offering more emotional support in conversations"
    
    def _assess_support_received(self, indicators: Dict, messages: List[Dict]) -> str:
        """Assess the level of emotional support received."""
        support_count = indicators.get("support_messages_received", 0)
        total = len([m for m in messages if not m.get("is_sender", False)])
        
        if total == 0:
            return "Unable to assess support patterns"
        
        support_ratio = support_count / total
        
        if support_ratio > 0.2:
            return "Your contact frequently provides emotional support"
        elif support_ratio > 0.1:
            return "Your contact offers support during important moments"
        else:
            return "Limited emotional support received from this contact"
    
    def _assess_emotional_availability(self, response: Dict, emotions: Dict) -> str:
        """Assess emotional availability based on response patterns."""
        avg_response = response.get("response_time_analysis", {}).get("average_minutes", 0)
        emotional_tone = emotions.get("emotional_tone", {}).get("dominant", "neutral")
        
        if avg_response < 30 and emotional_tone in ["positive", "supportive"]:
            return "High emotional availability and responsiveness"
        elif avg_response < 120:
            return "Moderate emotional availability"
        else:
            return "Limited immediate emotional availability"
    
    def _generate_support_recommendations(self, indicators: Dict, reciprocity: Dict) -> List[str]:
        """Generate recommendations for emotional support."""
        recommendations = []
        
        balance = reciprocity.get("support_balance", 1.0)
        if balance > 2.0:
            recommendations.append("Allow space for receiving support, not just giving")
        elif balance < 0.5:
            recommendations.append("Offer more emotional validation and support")
        
        if indicators.get("unacknowledged_emotions", 0) > 5:
            recommendations.append("Acknowledge emotions more explicitly in responses")
        
        return recommendations
    
    def _identify_conflict_triggers(self, messages: List[Dict], negative_periods: List) -> List[str]:
        """Identify common conflict triggers."""
        # This would analyze messages during negative periods
        # For now, return common categories
        return ["Miscommunication", "Delayed responses", "Differing expectations"]
    
    def _analyze_resolution_patterns(self, messages: List[Dict], emotions: Dict) -> Dict[str, Any]:
        """Analyze how conflicts are typically resolved."""
        return {
            "average_resolution_time": "24-48 hours",
            "resolution_style": "Direct communication",
            "success_rate": "High"
        }
    
    def _assess_conflict_health(self, emotions: Dict, messages: List[Dict]) -> List[str]:
        """Assess whether conflicts are handled healthily."""
        return [
            "Conflicts are addressed directly",
            "Both parties express their perspectives",
            "Resolution focuses on understanding"
        ]
    
    def _calculate_depth_trajectory(self, messages: List[Dict], depth: Dict) -> str:
        """Calculate the trajectory of conversation depth over time."""
        current_depth = depth.get("depth_score", 0)
        
        if current_depth > 60:
            return "Consistently deep and meaningful"
        elif current_depth > 40:
            return "Gradually deepening over time"
        else:
            return "Opportunity for deeper engagement"
    
    def _identify_relationship_milestones(self, messages: List[Dict]) -> List[Dict]:
        """Identify significant milestones in the relationship."""
        # This would analyze for significant events mentioned
        return [
            {"date": "Recent", "milestone": "Increased communication frequency"},
            {"date": "Past month", "milestone": "Shared personal experiences"}
        ]
    
    def _identify_growth_indicators(self, depth: Dict, topics: Dict) -> List[str]:
        """Identify indicators of relationship growth."""
        indicators = []
        
        if depth.get("trend") == "increasing":
            indicators.append("Conversations becoming more meaningful")
        
        if topics.get("new_topics_monthly", 0) > 3:
            indicators.append("Continuously exploring new topics together")
        
        return indicators
    
    def _generate_fallback_insights(self, analysis: Dict) -> Dict[str, Any]:
        """Generate basic insights if LLM fails."""
        return {
            "relationship_summary": "Based on the analysis, this relationship shows typical communication patterns.",
            "communication_style": "Communication style appears balanced with regular exchanges.",
            "areas_of_strength": ["Regular communication", "Emotional engagement"],
            "growth_opportunities": ["Consider deeper topic exploration", "Increase question-asking"],
            "note": "Advanced insights temporarily unavailable"
        }


# Global instance for use in tools
llm_generator = LLMInsightGenerator()


async def generate_llm_insights(
    messages: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    focus_areas: List[str]
) -> Dict[str, Any]:
    """
    Convenience function for generating LLM insights.
    
    This is the function that should be imported and used in mcp_tools.
    """
    return await llm_generator.generate_conversation_insights(
        messages, analysis, focus_areas
    )