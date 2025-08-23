"""
Memory monitoring utilities for the MCP Server application.

This module provides memory monitoring capabilities to prevent out-of-memory
issues. It includes a MemoryMonitor class that can track memory usage and
trigger callbacks when memory usage exceeds certain thresholds.
"""

import asyncio
import gc
import logging
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional

from ..exceptions import MemoryLimitError

logger = logging.getLogger(__name__)

# Import psutil if available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed, memory monitoring will be limited")


class MemoryMonitor:
    """
    Memory monitor for preventing out-of-memory situations.
    
    This class implements the Singleton pattern to ensure only one
    instance exists throughout the application. It provides memory
    usage tracking and can trigger callbacks when memory thresholds
    are exceeded.
    """
    
    # Singleton instance
    _instance = None
    
    @classmethod
    async def get_instance(cls) -> 'MemoryMonitor':
        """Get the singleton instance of the MemoryMonitor."""
        if cls._instance is None:
            cls._instance = MemoryMonitor()
            await cls._instance.start()
        return cls._instance
    
    def __init__(self):
        """Initialize the MemoryMonitor."""
        # Prevent multiple instances
        if MemoryMonitor._instance is not None:
            raise RuntimeError("MemoryMonitor is a singleton. Use get_instance() instead.")
        
        # Memory thresholds as a percentage of the memory limit
        self.high_threshold = 0.75  # 75% of memory limit
        self.low_threshold = 0.5    # 50% of memory limit
        
        # Memory limit in bytes (2GB by default)
        self.memory_limit_bytes = 2 * 1024 * 1024 * 1024
        
        # Current memory usage
        self.process_memory_bytes = 0
        self.process_memory_mb = 0
        self.process_memory_percent = 0
        self.system_memory_percent = 0
        
        # Tracking state
        self.is_monitoring = False
        self.is_in_high_memory_state = False
        self.monitoring_task = None
        
        # Callbacks for memory events
        self.high_memory_callbacks: List[Callable[[float, Dict[str, Any]], None]] = []
        self.low_memory_callbacks: List[Callable[[float, Dict[str, Any]], None]] = []
        self.emergency_callbacks: List[Callable[[float, Dict[str, Any]], None]] = []
        
        # Monitor interval in seconds
        self.monitor_interval = 5
        
        # Save as singleton instance
        MemoryMonitor._instance = self
    
    async def start(self) -> bool:
        """
        Start memory monitoring.
        
        Returns:
            True if monitoring was started successfully
        """
        if self.is_monitoring:
            logger.warning("Memory monitoring is already running")
            return True
        
        if not PSUTIL_AVAILABLE:
            logger.warning("Memory monitoring requires psutil. Install with 'pip install psutil'")
            return False
        
        try:
            # Start the monitoring task
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._monitor_memory())
            
            logger.info("Memory monitoring started")
            return True
        except Exception as e:
            logger.error(f"Error starting memory monitoring: {e}")
            logger.error(traceback.format_exc())
            self.is_monitoring = False
            return False
    
    async def stop(self) -> None:
        """Stop memory monitoring."""
        if not self.is_monitoring:
            return
        
        try:
            # Cancel the monitoring task
            if self.monitoring_task:
                self.monitoring_task.cancel()
                try:
                    await self.monitoring_task
                except asyncio.CancelledError:
                    pass
            
            self.is_monitoring = False
            logger.info("Memory monitoring stopped")
        except Exception as e:
            logger.error(f"Error stopping memory monitoring: {e}")
            logger.error(traceback.format_exc())
    
    async def _monitor_memory(self) -> None:
        """Background task to monitor memory usage."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            process = psutil.Process()
            
            while self.is_monitoring:
                try:
                    # Get memory information
                    process_memory = process.memory_info()
                    system_memory = psutil.virtual_memory()
                    
                    # Update memory usage statistics
                    self.process_memory_bytes = process_memory.rss
                    self.process_memory_mb = self.process_memory_bytes / (1024 * 1024)
                    self.process_memory_percent = (self.process_memory_bytes / self.memory_limit_bytes) * 100
                    self.system_memory_percent = system_memory.percent
                    
                    # Check memory thresholds
                    await self._check_memory_thresholds(process_memory, system_memory)
                    
                    # Wait before checking again
                    await asyncio.sleep(self.monitor_interval)
                except asyncio.CancelledError:
                    logger.info("Memory monitoring task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                    logger.error(traceback.format_exc())
                    # Continue monitoring despite errors
                    await asyncio.sleep(self.monitor_interval)
        except Exception as e:
            logger.error(f"Fatal error in memory monitoring: {e}")
            logger.error(traceback.format_exc())
            self.is_monitoring = False
    
    async def _check_memory_thresholds(
        self, process_memory: Any, system_memory: Any
    ) -> None:
        """
        Check memory thresholds and trigger callbacks if needed.
        
        Args:
            process_memory: Process memory information from psutil
            system_memory: System memory information from psutil
        """
        # Calculate percentage of memory limit
        memory_percent = (process_memory.rss / self.memory_limit_bytes) * 100
        
        # Log memory usage at different log levels based on usage
        if memory_percent > 90:
            logger.critical(
                f"Critical memory usage: {memory_percent:.1f}% "
                f"({self.process_memory_mb:.1f}MB / "
                f"{self.memory_limit_bytes/(1024*1024):.1f}MB)"
            )
        elif memory_percent > 75:
            logger.warning(
                f"High memory usage: {memory_percent:.1f}% "
                f"({self.process_memory_mb:.1f}MB / "
                f"{self.memory_limit_bytes/(1024*1024):.1f}MB)"
            )
        else:
            logger.debug(
                f"Current memory usage: {memory_percent:.1f}% "
                f"({self.process_memory_mb:.1f}MB / "
                f"{self.memory_limit_bytes/(1024*1024):.1f}MB)"
            )
        
        # System memory info for context
        system_info = {
            "process": {
                "rss_bytes": process_memory.rss,
                "rss_mb": process_memory.rss / (1024 * 1024),
                "percent": memory_percent,
            },
            "system": {
                "total_mb": system_memory.total / (1024 * 1024),
                "available_mb": system_memory.available / (1024 * 1024),
                "percent": system_memory.percent,
            }
        }
        
        # Check emergency threshold (95%)
        if memory_percent > 95:
            logger.critical("Memory usage exceeds emergency threshold (95%)")
            # Trigger emergency callbacks
            for callback in self.emergency_callbacks:
                try:
                    await callback(memory_percent, system_info)
                except Exception as e:
                    logger.error(f"Error in emergency memory callback: {e}")
            
            # Force garbage collection
            gc.collect()
        
        # Check high memory threshold
        elif memory_percent > self.high_threshold * 100 and not self.is_in_high_memory_state:
            logger.warning(f"Memory usage exceeds high threshold ({self.high_threshold*100}%)")
            self.is_in_high_memory_state = True
            
            # Trigger high memory callbacks
            for callback in self.high_memory_callbacks:
                try:
                    await callback(memory_percent, system_info)
                except Exception as e:
                    logger.error(f"Error in high memory callback: {e}")
        
        # Check low memory threshold (for recovery)
        elif memory_percent < self.low_threshold * 100 and self.is_in_high_memory_state:
            logger.info(f"Memory usage below low threshold ({self.low_threshold*100}%)")
            self.is_in_high_memory_state = False
            
            # Trigger low memory callbacks
            for callback in self.low_memory_callbacks:
                try:
                    await callback(memory_percent, system_info)
                except Exception as e:
                    logger.error(f"Error in low memory callback: {e}")
    
    def register_high_memory_callback(
        self, callback: Callable[[float, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback to be called when high memory threshold is exceeded.
        
        Args:
            callback: Function to call with memory percent and system info
        """
        self.high_memory_callbacks.append(callback)
    
    def register_low_memory_callback(
        self, callback: Callable[[float, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback to be called when memory usage drops below low threshold.
        
        Args:
            callback: Function to call with memory percent and system info
        """
        self.low_memory_callbacks.append(callback)
    
    def register_emergency_callback(
        self, callback: Callable[[float, Dict[str, Any]], None]
    ) -> None:
        """
        Register a callback to be called in memory emergency situations.
        
        Args:
            callback: Function to call with memory percent and system info
        """
        self.emergency_callbacks.append(callback)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory statistics.
        
        Returns:
            Dict with memory statistics
        """
        return {
            "process": {
                "rss_bytes": self.process_memory_bytes,
                "rss_mb": self.process_memory_mb,
                "percent": self.process_memory_percent,
                "limit_mb": self.memory_limit_bytes / (1024 * 1024),
            },
            "system": {
                "percent": self.system_memory_percent,
            },
            "thresholds": {
                "high": self.high_threshold * 100,
                "low": self.low_threshold * 100,
            }
        }


# Convenience functions for modules that don't want to use the singleton

async def initialize_memory_monitor() -> bool:
    """
    Initialize and start the memory monitor.
    
    Returns:
        True if monitor was successfully started
    """
    if not PSUTIL_AVAILABLE:
        logger.warning("Memory monitoring requires psutil. Install with 'pip install psutil'")
        return False
    
    monitor = await MemoryMonitor.get_instance()
    return await monitor.start()


async def limit_memory(limit_bytes: int) -> bool:
    """
    Set the memory limit for monitoring.
    
    Args:
        limit_bytes: Memory limit in bytes
        
    Returns:
        True if limit was successfully set
    """
    try:
        monitor = await MemoryMonitor.get_instance()
        monitor.memory_limit_bytes = limit_bytes
        logger.info(f"Memory limit set to {limit_bytes / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        logger.error(f"Error setting memory limit: {e}")
        return False


async def stop_memory_monitoring() -> None:
    """Stop memory monitoring."""
    if not PSUTIL_AVAILABLE:
        return
    
    if MemoryMonitor._instance:
        await MemoryMonitor._instance.stop()
