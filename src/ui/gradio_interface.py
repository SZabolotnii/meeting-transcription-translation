"""
Gradio user interface for the Meeting Transcription and Translation System.
Provides web-based interface for controlling live subtitles with translation.
"""

import gradio as gr
import asyncio
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json

from ..config import config, SUPPORTED_LANGUAGES, AUDIO_SOURCE_TYPES


class GradioInterface:
    """Main Gradio interface for the transcription system"""
    
    def __init__(self):
        self.session_active = False
        self.current_subtitles = ""
        self.session_history = []
        self.status_message = "Готовий до роботи"
        self.audio_level = 0.0
        self.session_start_time = None
        self.current_audio_source = None
        self.current_target_language = None
        self.current_audio_device = None
        self.settings = {}
        
        # Processing status tracking
        self.processing_status = {
            'audio_capture': 'idle',
            'transcription': 'idle', 
            'translation': 'idle'
        }
        self.error_count = 0
        self.last_error = None
        
        # Performance metrics
        self.performance_metrics = {
            'transcription_latency': 0.0,
            'translation_latency': 0.0,
            'total_processed': 0,
            'errors_count': 0
        }
        
        # Initialize interface components
        self.interface = None
        self._components = {}
        
    def create_interface(self) -> gr.Interface:
        """Create and configure the Gradio interface"""
        
        with gr.Blocks(
            title="Система Транскрибування та Перекладу Мітингів",
            theme=gr.themes.Soft(),
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown("# 🎤 Система Транскрибування та Перекладу Мітингів")
            gr.Markdown("Живі субтитри з автоматичним перекладом в режимі реального часу")
            
            with gr.Row():
                # Left column - Controls
                with gr.Column(scale=1):
                    gr.Markdown("## ⚙️ Налаштування")
                    
                    # Audio source selection
                    audio_source = gr.Radio(
                        choices=list(AUDIO_SOURCE_TYPES.values()),
                        value="Мікрофон",
                        label="Джерело аудіо",
                        info="Оберіть джерело для захоплення звуку"
                    )
                    
                    # Audio device selection (will be populated dynamically)
                    audio_device = gr.Dropdown(
                        choices=["Пристрій за замовчуванням"],
                        value="Пристрій за замовчуванням",
                        label="Аудіо пристрій",
                        info="Оберіть конкретний аудіо пристрій",
                        interactive=True
                    )
                    
                    # Refresh devices button
                    refresh_devices_btn = gr.Button(
                        "🔄 Оновити пристрої",
                        variant="secondary",
                        size="sm"
                    )
                    
                    # Target language selection
                    target_language = gr.Dropdown(
                        choices=list(SUPPORTED_LANGUAGES.values()),
                        value="Українська",
                        label="Мова перекладу",
                        info="Мова для відображення субтитрів"
                    )
                    
                    # Control buttons
                    with gr.Row():
                        start_btn = gr.Button(
                            "▶️ Старт",
                            variant="primary",
                            size="lg"
                        )
                        stop_btn = gr.Button(
                            "⏹️ Стоп",
                            variant="secondary",
                            size="lg",
                            interactive=False
                        )
                    
                    # Status section
                    gr.Markdown("## 📊 Статус Системи")
                    
                    # Main system status
                    status_indicator = gr.HTML(
                        value=self._format_status_html("Готовий до роботи", False),
                        label="Статус системи"
                    )
                    
                    # Processing status indicator
                    processing_status = gr.HTML(
                        value=self._format_processing_status_html(),
                        label="Статус обробки"
                    )
                    
                    # Audio level indicator with visual meter
                    audio_level_display = gr.HTML(
                        value=self._format_audio_level_html(0.0),
                        label="Рівень аудіо"
                    )
                    
                    # Performance metrics
                    performance_metrics = gr.HTML(
                        value=self._format_performance_metrics_html(),
                        label="Метрики продуктивності"
                    )
                    
                    # Export section
                    gr.Markdown("## 💾 Експорт та Налаштування")
                    
                    # Export format selection
                    export_format = gr.Radio(
                        choices=["Текст (.txt)", "JSON (.json)", "SRT Субтитри (.srt)"],
                        value="Текст (.txt)",
                        label="Формат експорту",
                        info="Оберіть формат для збереження історії"
                    )
                    
                    # Export options
                    with gr.Row():
                        include_timestamps = gr.Checkbox(
                            value=True,
                            label="Включити часові мітки"
                        )
                        include_original = gr.Checkbox(
                            value=True,
                            label="Включити оригінальний текст"
                        )
                    
                    with gr.Row():
                        include_translation = gr.Checkbox(
                            value=True,
                            label="Включити переклад"
                        )
                        separate_files = gr.Checkbox(
                            value=False,
                            label="Окремі файли для мов"
                        )
                    
                    # Export buttons
                    with gr.Row():
                        export_btn = gr.Button(
                            "📥 Завантажити історію",
                            variant="secondary",
                            interactive=False
                        )
                        clear_history_btn = gr.Button(
                            "🗑️ Очистити історію",
                            variant="secondary",
                            interactive=False
                        )
                    
                    export_file = gr.File(
                        label="Файл історії субтитрів",
                        visible=False
                    )
                    
                    # Advanced settings
                    with gr.Accordion("⚙️ Розширені налаштування", open=False):
                        # Buffer settings
                        buffer_duration = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=config.audio.buffer_duration,
                            step=0.5,
                            label="Тривалість буфера (сек)",
                            info="Розмір аудіо сегментів для обробки"
                        )
                        
                        # Auto-refresh interval
                        refresh_interval = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=config.ui.auto_refresh_interval,
                            step=0.1,
                            label="Інтервал оновлення (сек)",
                            info="Частота оновлення інтерфейсу"
                        )
                        
                        # Max history entries
                        max_history = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=100,
                            step=10,
                            label="Максимум записів в історії",
                            info="Кількість субтитрів для відображення"
                        )
                        
                        # Apply settings button
                        apply_settings_btn = gr.Button(
                            "✅ Застосувати налаштування",
                            variant="primary"
                        )
                
                # Right column - Live subtitles
                with gr.Column(scale=2):
                    gr.Markdown("## 📝 Живі Субтитри")
                    
                    live_subtitles = gr.Textbox(
                        value="",
                        label="",
                        placeholder="Субтитри з'являться тут після запуску сесії...",
                        lines=20,
                        max_lines=25,
                        interactive=False,
                        show_label=False,
                        container=True,
                        elem_classes=["subtitles-display"]
                    )
                    
                    # Session info
                    session_info = gr.HTML(
                        value=self._format_session_info_html(),
                        label="Інформація про сесію"
                    )
            
            # Auto-refresh for live updates
            # Set up periodic refresh when session is active
            refresh_timer = gr.Timer(
                value=config.ui.auto_refresh_interval,
                active=False
            )
            
            # Event handlers
            start_btn.click(
                fn=self._start_session,
                inputs=[audio_source, target_language, audio_device],
                outputs=[start_btn, stop_btn, status_indicator, export_btn, refresh_timer, processing_status, performance_metrics, clear_history_btn]
            )
            
            stop_btn.click(
                fn=self._stop_session,
                inputs=[],
                outputs=[start_btn, stop_btn, status_indicator, export_btn, refresh_timer, processing_status, performance_metrics, clear_history_btn]
            )
            
            # Export and settings event handlers
            export_btn.click(
                fn=self._export_history,
                inputs=[export_format, include_timestamps, include_original, include_translation, separate_files],
                outputs=[export_file]
            )
            
            clear_history_btn.click(
                fn=self._clear_history,
                inputs=[],
                outputs=[clear_history_btn, export_btn]
            )
            
            refresh_devices_btn.click(
                fn=self._refresh_audio_devices,
                inputs=[],
                outputs=[audio_device]
            )
            
            apply_settings_btn.click(
                fn=self._apply_settings,
                inputs=[buffer_duration, refresh_interval, max_history],
                outputs=[apply_settings_btn]
            )
            
            # Audio source change handler
            audio_source.change(
                fn=self._on_audio_source_change,
                inputs=[audio_source],
                outputs=[audio_device]
            )
            
            # Live update event handler
            refresh_timer.tick(
                fn=self._update_live_content,
                inputs=[],
                outputs=[live_subtitles, session_info, audio_level_display, status_indicator, processing_status, performance_metrics]
            )
            
            # Store components for later access
            self._components = {
                'live_subtitles': live_subtitles,
                'session_info': session_info,
                'audio_level_display': audio_level_display,
                'status_indicator': status_indicator,
                'refresh_timer': refresh_timer,
                'start_btn': start_btn,
                'stop_btn': stop_btn,
                'export_btn': export_btn
            }
            
        self.interface = interface
        return interface
    
    def _start_session(self, audio_source: str, target_language: str, audio_device: str) -> Tuple:
        """Start transcription session"""
        self.session_active = True
        self.status_message = "Сесія активна"
        self.session_history = []
        self.current_subtitles = ""
        self.session_start_time = datetime.now()
        
        # Store current session settings
        self.current_audio_source = audio_source
        self.current_target_language = target_language
        self.current_audio_device = audio_device
        
        # Reset performance metrics
        self.performance_metrics = {
            'transcription_latency': 0.0,
            'translation_latency': 0.0,
            'total_processed': 0,
            'errors_count': 0
        }
        self.error_count = 0
        self.last_error = None
        
        # TODO: Initialize orchestrator and start transcription
        # This will be connected to the orchestrator in later tasks
        
        return (
            gr.Button(interactive=False),  # start_btn
            gr.Button(interactive=True),   # stop_btn
            self._format_status_html("Сесія активна", True),  # status
            gr.Button(interactive=False),  # export_btn
            gr.Timer(active=True),  # refresh_timer - start auto-refresh
            self._format_processing_status_html(),  # processing_status
            self._format_performance_metrics_html(),  # performance_metrics
            gr.Button(interactive=True)  # clear_history_btn
        )
    
    def _stop_session(self) -> Tuple:
        """Stop transcription session"""
        self.session_active = False
        self.status_message = "Сесія зупинена"
        
        # TODO: Stop orchestrator
        # This will be connected to the orchestrator in later tasks
        
        return (
            gr.Button(interactive=True),   # start_btn
            gr.Button(interactive=False),  # stop_btn
            self._format_status_html("Сесія зупинена", False),  # status
            gr.Button(interactive=True) if self.session_history else gr.Button(interactive=False),  # export_btn
            gr.Timer(active=False),  # refresh_timer - stop auto-refresh
            self._format_processing_status_html(),  # processing_status
            self._format_performance_metrics_html(),  # performance_metrics
            gr.Button(interactive=True) if self.session_history else gr.Button(interactive=False)  # clear_history_btn
        )
    
    def _export_history(self, export_format: str, include_timestamps: bool, 
                       include_original: bool, include_translation: bool, 
                       separate_files: bool) -> Optional[str]:
        """Export session history with specified format and options"""
        if not self.session_history:
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine file extension based on format
        if "JSON" in export_format:
            extension = "json"
        elif "SRT" in export_format:
            extension = "srt"
        else:
            extension = "txt"
        
        filename = f"subtitles_history_{timestamp}.{extension}"
        
        # Generate content based on format
        if "JSON" in export_format:
            content = self._generate_json_export(include_timestamps, include_original, include_translation)
        elif "SRT" in export_format:
            content = self._generate_srt_export(include_timestamps, include_original, include_translation)
        else:
            content = self._generate_text_export(include_timestamps, include_original, include_translation)
        
        # Save to temporary file
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def _generate_text_export(self, include_timestamps: bool, include_original: bool, include_translation: bool) -> str:
        """Generate text format export"""
        content = f"Історія субтитрів - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 60 + "\n\n"
        
        if self.current_audio_source and self.current_target_language:
            content += f"Джерело аудіо: {self.current_audio_source}\n"
            content += f"Мова перекладу: {self.current_target_language}\n"
            content += f"Всього записів: {len(self.session_history)}\n\n"
        
        for i, entry in enumerate(self.session_history, 1):
            content += f"--- Запис {i} ---\n"
            
            if include_timestamps:
                content += f"Час: {entry['timestamp']}\n"
            
            if include_original and entry.get('original_text'):
                content += f"Оригінал: {entry['original_text']}\n"
            
            if include_translation and entry.get('translated_text'):
                content += f"Переклад: {entry['translated_text']}\n"
            
            content += "\n"
        
        return content
    
    def _generate_json_export(self, include_timestamps: bool, include_original: bool, include_translation: bool) -> str:
        """Generate JSON format export"""
        import json
        
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "audio_source": self.current_audio_source,
                "target_language": self.current_target_language,
                "total_entries": len(self.session_history)
            },
            "subtitles": []
        }
        
        for entry in self.session_history:
            subtitle_entry = {}
            
            if include_timestamps:
                subtitle_entry["timestamp"] = entry["timestamp"]
            
            if include_original and entry.get('original_text'):
                subtitle_entry["original_text"] = entry["original_text"]
            
            if include_translation and entry.get('translated_text'):
                subtitle_entry["translated_text"] = entry["translated_text"]
            
            export_data["subtitles"].append(subtitle_entry)
        
        return json.dumps(export_data, ensure_ascii=False, indent=2)
    
    def _generate_srt_export(self, include_timestamps: bool, include_original: bool, include_translation: bool) -> str:
        """Generate SRT subtitle format export"""
        content = ""
        
        for i, entry in enumerate(self.session_history, 1):
            # SRT format: sequence number, timecode, subtitle text, blank line
            content += f"{i}\n"
            
            # Generate timecode (simplified - using entry index for timing)
            start_seconds = (i - 1) * 3  # Assume 3 seconds per subtitle
            end_seconds = i * 3
            
            start_time = self._seconds_to_srt_time(start_seconds)
            end_time = self._seconds_to_srt_time(end_seconds)
            
            content += f"{start_time} --> {end_time}\n"
            
            # Add subtitle text
            if include_original and entry.get('original_text'):
                content += f"{entry['original_text']}\n"
            
            if include_translation and entry.get('translated_text'):
                if include_original:
                    content += f"[{entry['translated_text']}]\n"
                else:
                    content += f"{entry['translated_text']}\n"
            
            content += "\n"
        
        return content
    
    def _seconds_to_srt_time(self, seconds: int) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d},000"
    
    def _clear_history(self) -> Tuple:
        """Clear session history"""
        self.session_history = []
        self.current_subtitles = ""
        
        return (
            gr.Button(interactive=False),  # clear_history_btn
            gr.Button(interactive=False)   # export_btn
        )
    
    def _refresh_audio_devices(self) -> gr.Dropdown:
        """Refresh available audio devices"""
        # TODO: Implement actual audio device detection
        # For now, return mock devices
        devices = [
            "Пристрій за замовчуванням",
            "MacBook Pro Microphone",
            "AirPods Pro",
            "BlackHole 2ch",
            "External USB Microphone"
        ]
        
        return gr.Dropdown(choices=devices, value=devices[0])
    
    def _on_audio_source_change(self, audio_source: str) -> gr.Dropdown:
        """Handle audio source change"""
        if audio_source == "Системне аудіо":
            # Show system audio devices
            devices = [
                "BlackHole 2ch",
                "Soundflower (2ch)",
                "Loopback Audio"
            ]
        else:
            # Show microphone devices
            devices = [
                "Пристрій за замовчуванням",
                "MacBook Pro Microphone", 
                "AirPods Pro",
                "External USB Microphone"
            ]
        
        return gr.Dropdown(choices=devices, value=devices[0])
    
    def _apply_settings(self, buffer_duration: float, refresh_interval: float, max_history: int) -> gr.Button:
        """Apply advanced settings"""
        # TODO: Apply settings to the system
        # For now, just store them
        self.settings = {
            'buffer_duration': buffer_duration,
            'refresh_interval': refresh_interval,
            'max_history': max_history
        }
        
        # In a real implementation, these would update the actual system configuration
        return gr.Button(value="✅ Налаштування застосовано", variant="secondary")
    
    def _format_status_html(self, message: str, active: bool) -> str:
        """Format status indicator HTML"""
        color = "#28a745" if active else "#6c757d"
        icon = "🟢" if active else "⚫"
        
        return f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid {color};">
            <strong>{icon} {message}</strong>
        </div>
        """
    
    def _format_audio_level_html(self, level: float) -> str:
        """Format audio level indicator HTML"""
        # Convert level to percentage (0-100)
        percentage = min(int(level * 100), 100)
        
        # Determine color and status based on level
        if percentage < 10:
            color = "#dc3545"  # Red - too low
            status = "Дуже тихо"
            icon = "🔇"
        elif percentage < 30:
            color = "#fd7e14"  # Orange - low
            status = "Тихо"
            icon = "🔉"
        elif percentage < 80:
            color = "#28a745"  # Green - good
            status = "Оптимально"
            icon = "🔊"
        else:
            color = "#ffc107"  # Yellow - might be too high
            status = "Гучно"
            icon = "📢"
        
        return f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid {color};">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.2em;">{icon}</span>
                <span style="font-weight: bold;">Рівень аудіо: {status}</span>
                <span style="font-weight: bold; color: {color};">{percentage}%</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 10px; height: 8px; overflow: hidden;">
                <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, {color} 0%, {color} 100%); transition: width 0.3s ease;"></div>
            </div>
        </div>
        """
    
    def _format_processing_status_html(self) -> str:
        """Format processing status indicator HTML"""
        if not self.session_active:
            return """
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px; text-align: center; color: #6c757d;">
                <em>Обробка не активна</em>
            </div>
            """
        
        # Status icons and colors
        status_config = {
            'idle': {'icon': '⚫', 'color': '#6c757d', 'text': 'Очікування'},
            'active': {'icon': '🟢', 'color': '#28a745', 'text': 'Активно'},
            'processing': {'icon': '🟡', 'color': '#ffc107', 'text': 'Обробка'},
            'error': {'icon': '🔴', 'color': '#dc3545', 'text': 'Помилка'}
        }
        
        components = [
            ('🎤 Захоплення аудіо', self.processing_status.get('audio_capture', 'idle')),
            ('🎯 Транскрибування', self.processing_status.get('transcription', 'idle')),
            ('🌐 Переклад', self.processing_status.get('translation', 'idle'))
        ]
        
        status_html = []
        for name, status in components:
            config = status_config.get(status, status_config['idle'])
            status_html.append(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin: 5px 0;">
                    <span>{name}</span>
                    <span style="color: {config['color']}; font-weight: bold;">
                        {config['icon']} {config['text']}
                    </span>
                </div>
            """)
        
        return f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid #007bff;">
            <div style="font-weight: bold; margin-bottom: 10px; color: #007bff;">
                ⚙️ Статус Компонентів
            </div>
            {''.join(status_html)}
        </div>
        """
    
    def _format_performance_metrics_html(self) -> str:
        """Format performance metrics HTML"""
        if not self.session_active:
            return """
            <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px; text-align: center; color: #6c757d;">
                <em>Метрики недоступні</em>
            </div>
            """
        
        # Calculate total latency
        total_latency = self.performance_metrics['transcription_latency'] + self.performance_metrics['translation_latency']
        
        # Determine latency status
        if total_latency < 2.0:
            latency_color = "#28a745"  # Green - excellent
            latency_status = "Відмінно"
        elif total_latency < 4.0:
            latency_color = "#ffc107"  # Yellow - good
            latency_status = "Добре"
        else:
            latency_color = "#dc3545"  # Red - needs attention
            latency_status = "Потребує уваги"
        
        return f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 8px; border-left: 4px solid {latency_color};">
            <div style="font-weight: bold; margin-bottom: 10px; color: {latency_color};">
                📊 Метрики Продуктивності
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.9em;">
                <div>
                    <strong>Затримка:</strong><br>
                    <span style="color: {latency_color};">{total_latency:.1f}с ({latency_status})</span>
                </div>
                <div>
                    <strong>Оброблено:</strong><br>
                    <span style="color: #007bff;">{self.performance_metrics['total_processed']}</span>
                </div>
                <div>
                    <strong>Транскрибування:</strong><br>
                    <span>{self.performance_metrics['transcription_latency']:.1f}с</span>
                </div>
                <div>
                    <strong>Переклад:</strong><br>
                    <span>{self.performance_metrics['translation_latency']:.1f}с</span>
                </div>
            </div>
            {self._format_error_info() if self.error_count > 0 else ''}
        </div>
        """
    
    def _format_error_info(self) -> str:
        """Format error information HTML"""
        if self.error_count == 0:
            return ""
        
        return f"""
        <div style="margin-top: 10px; padding: 8px; background-color: #f8d7da; border-radius: 4px; border-left: 3px solid #dc3545;">
            <div style="font-weight: bold; color: #721c24; font-size: 0.9em;">
                ⚠️ Помилки: {self.error_count}
            </div>
            {f'<div style="font-size: 0.8em; color: #721c24; margin-top: 4px;">{self.last_error}</div>' if self.last_error else ''}
        </div>
        """
    
    def _simulate_processing_activity(self):
        """Simulate processing activity for demonstration"""
        import random
        
        # Randomly update processing status
        components = ['audio_capture', 'transcription', 'translation']
        statuses = ['idle', 'active', 'processing']
        
        for component in components:
            if random.random() < 0.3:  # 30% chance to change status
                self.processing_status[component] = random.choice(statuses)
        
        # Occasionally simulate errors
        if random.random() < 0.05:  # 5% chance of error
            self.error_count += 1
            self.last_error = "Тимчасова помилка мережі"
            self.performance_metrics['errors_count'] += 1
    
    def _update_live_content(self) -> Tuple:
        """Update live content (subtitles, session info, audio level, status)"""
        if not self.session_active:
            return (
                self.current_subtitles,
                self._format_session_info_html(),
                self._format_audio_level_html(0.0),
                self._format_status_html("Сесія не активна", False),
                self._format_processing_status_html(),
                self._format_performance_metrics_html()
            )
        
        # TODO: Get real-time data from orchestrator
        # For now, simulate some activity for demonstration
        import random
        import time
        
        # Simulate audio level changes
        self.audio_level = random.uniform(0.3, 0.8)
        
        # Simulate processing status changes
        self._simulate_processing_activity()
        
        # Simulate new subtitles (this will be replaced with real data)
        if random.random() < 0.1:  # 10% chance of new subtitle
            current_time = datetime.now().strftime("%H:%M:%S")
            new_subtitle = f"Демонстраційний текст субтитрів..."
            
            # Simulate processing latency
            transcription_latency = random.uniform(0.5, 2.0)
            translation_latency = random.uniform(0.2, 1.0)
            
            # Update performance metrics
            self.performance_metrics['transcription_latency'] = transcription_latency
            self.performance_metrics['translation_latency'] = translation_latency
            self.performance_metrics['total_processed'] += 1
            
            # Add to history
            self.session_history.append({
                'timestamp': current_time,
                'original_text': new_subtitle,
                'translated_text': f"Переклад: {new_subtitle}"
            })
            
            # Update current subtitles with auto-scroll effect
            self._update_subtitles_display()
        
        return (
            self.current_subtitles,
            self._format_session_info_html(),
            self._format_audio_level_html(self.audio_level),
            self._format_status_html("Сесія активна", True),
            self._format_processing_status_html(),
            self._format_performance_metrics_html()
        )
    
    def _update_subtitles_display(self):
        """Update the subtitles display with automatic scrolling"""
        # Keep only the last 50 entries for performance
        recent_history = self.session_history[-50:]
        
        # Format subtitles for display with better formatting
        subtitle_lines = []
        for i, entry in enumerate(recent_history):
            # Add timestamp and original text
            subtitle_lines.append(f"🕐 [{entry['timestamp']}]")
            subtitle_lines.append(f"   {entry['original_text']}")
            
            # Add translation if available
            if entry.get('translated_text'):
                subtitle_lines.append(f"   🔄 {entry['translated_text']}")
            
            # Add separator between entries (except for the last one)
            if i < len(recent_history) - 1:
                subtitle_lines.append("─" * 50)
            
            subtitle_lines.append("")  # Empty line for spacing
        
        # Add a marker at the end to ensure auto-scroll
        if subtitle_lines:
            subtitle_lines.append("📍 Кінець субтитрів")
        
        self.current_subtitles = "\n".join(subtitle_lines)
    
    def _format_session_info_html(self) -> str:
        """Format session information HTML"""
        if not self.session_active:
            return """
            <div style="padding: 10px; text-align: center; color: #6c757d; font-style: italic;">
                Сесія не активна
            </div>
            """
        
        subtitle_count = len(self.session_history)
        
        # Calculate session duration
        if self.session_start_time:
            duration = datetime.now() - self.session_start_time
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            session_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            session_duration = "00:00:00"
        
        return f"""
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span><strong>Субтитрів:</strong> {subtitle_count}</span>
                <span><strong>Тривалість:</strong> {session_duration}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: #6c757d;">
                <span>Джерело: {self.current_audio_source or 'Не обрано'}</span>
                <span>Мова: {self.current_target_language or 'Не обрано'}</span>
            </div>
        </div>
        """
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface"""
        return """
        .subtitles-display textarea {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            font-size: 16px !important;
            line-height: 1.5 !important;
            background-color: #f8f9fa !important;
            border: 2px solid #dee2e6 !important;
            border-radius: 8px !important;
            overflow-y: auto !important;
            scroll-behavior: smooth !important;
        }
        
        .subtitles-display textarea:focus {
            border-color: #007bff !important;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
        }
        
        /* Auto-scroll to bottom for new content */
        .subtitles-display textarea {
            resize: vertical !important;
        }
        
        /* Custom button styles */
        .gradio-button {
            border-radius: 6px !important;
            font-weight: 600 !important;
        }
        
        /* Status indicators */
        .status-indicator {
            margin: 10px 0;
        }
        
        /* Audio level indicator animation */
        .audio-level-bar {
            transition: width 0.3s ease !important;
        }
        
        /* Session info styling */
        .session-info {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            border: 1px solid #dee2e6 !important;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .subtitles-display textarea {
                font-size: 14px !important;
            }
        }
        
        /* Smooth transitions for all elements */
        * {
            transition: all 0.2s ease !important;
        }
        """
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if not self.interface:
            self.interface = self.create_interface()
        
        # Use configuration settings
        launch_kwargs = {
            'server_name': config.ui.host,
            'server_port': config.ui.port,
            'share': config.ui.share,
            'debug': config.ui.debug,
            **kwargs
        }
        
        return self.interface.launch(**launch_kwargs)


def create_interface() -> GradioInterface:
    """Factory function to create the Gradio interface"""
    return GradioInterface()


# For backwards compatibility and direct usage
def launch_interface(**kwargs):
    """Launch the interface directly"""
    interface = create_interface()
    return interface.launch(**kwargs)