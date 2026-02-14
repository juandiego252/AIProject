-- ============================================================
-- ESQUEMA DE BASE DE DATOS PARA SISTEMA DE RECONOCIMIENTO FACIAL
-- Ejecutar en Supabase SQL Editor
-- ============================================================

-- Tabla principal de registros de acceso
CREATE TABLE IF NOT EXISTS access_logs (
    id BIGSERIAL PRIMARY KEY,
    person_name TEXT,
    confidence DECIMAL(10, 2),
    access_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_granted BOOLEAN NOT NULL DEFAULT FALSE,
    event_type TEXT NOT NULL, -- 'successful_access', 'failed_access', 'no_face_detected'
    failure_reason TEXT, -- 'unknown_person', 'low_confidence', 'no_face_detected', etc.
    image_path TEXT,
    face_image_base64 TEXT,
    additional_data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Tabla de sesiones de entrenamiento
CREATE TABLE IF NOT EXISTS training_sessions (
    id BIGSERIAL PRIMARY KEY,
    person_name TEXT NOT NULL,
    images_count INTEGER NOT NULL,
    model_type TEXT NOT NULL, -- 'LBPHFace', 'EigenFace', 'FisherFace'
    training_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para mejorar el rendimiento de consultas
CREATE INDEX IF NOT EXISTS idx_access_logs_timestamp ON access_logs(access_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_access_logs_person_name ON access_logs(person_name);
CREATE INDEX IF NOT EXISTS idx_access_logs_access_granted ON access_logs(access_granted);
CREATE INDEX IF NOT EXISTS idx_access_logs_event_type ON access_logs(event_type);
CREATE INDEX IF NOT EXISTS idx_training_sessions_person_name ON training_sessions(person_name);
CREATE INDEX IF NOT EXISTS idx_training_sessions_timestamp ON training_sessions(training_timestamp DESC);

-- Habilitar Row Level Security (RLS)
ALTER TABLE access_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_sessions ENABLE ROW LEVEL SECURITY;

-- Políticas de acceso (ajustar según tus necesidades de seguridad)
-- Estas políticas permiten insertar y leer a cualquier usuario autenticado
-- IMPORTANTE: Modifica estas políticas según tus requisitos de seguridad

-- Política para access_logs
DROP POLICY IF EXISTS "Enable insert for authenticated users" ON access_logs;
CREATE POLICY "Enable insert for authenticated users"
ON access_logs
FOR INSERT
TO authenticated
WITH CHECK (true);

DROP POLICY IF EXISTS "Enable read access for all users" ON access_logs;
CREATE POLICY "Enable read access for all users"
ON access_logs
FOR SELECT
TO authenticated
USING (true);

-- Si quieres permitir acceso público (sin autenticación), usa estas políticas en su lugar:
-- DROP POLICY IF EXISTS "Enable insert for all users" ON access_logs;
-- CREATE POLICY "Enable insert for all users"
-- ON access_logs
-- FOR INSERT
-- TO anon
-- WITH CHECK (true);

-- DROP POLICY IF EXISTS "Enable read access for all users" ON access_logs;
-- CREATE POLICY "Enable read access for all users"
-- ON access_logs
-- FOR SELECT
-- TO anon
-- USING (true);

-- Política para training_sessions
DROP POLICY IF EXISTS "Enable insert for authenticated users" ON training_sessions;
CREATE POLICY "Enable insert for authenticated users"
ON training_sessions
FOR INSERT
TO authenticated
WITH CHECK (true);

DROP POLICY IF EXISTS "Enable read access for all users" ON training_sessions;
CREATE POLICY "Enable read access for all users"
ON training_sessions
FOR SELECT
TO authenticated
USING (true);

-- ============================================================
-- VISTAS ÚTILES PARA ANÁLISIS
-- ============================================================

-- Vista de accesos recientes
CREATE OR REPLACE VIEW recent_access_attempts AS
SELECT 
    id,
    person_name,
    confidence,
    access_timestamp,
    access_granted,
    event_type,
    failure_reason
FROM access_logs
ORDER BY access_timestamp DESC
LIMIT 100;

-- Vista de estadísticas por persona
CREATE OR REPLACE VIEW person_statistics AS
SELECT 
    person_name,
    COUNT(*) as total_attempts,
    SUM(CASE WHEN access_granted THEN 1 ELSE 0 END) as successful_access,
    SUM(CASE WHEN NOT access_granted THEN 1 ELSE 0 END) as failed_access,
    AVG(CASE WHEN access_granted THEN confidence ELSE NULL END) as avg_confidence_success,
    MAX(access_timestamp) as last_seen
FROM access_logs
WHERE person_name IS NOT NULL
GROUP BY person_name
ORDER BY total_attempts DESC;

-- Vista de estadísticas diarias
CREATE OR REPLACE VIEW daily_statistics AS
SELECT 
    DATE(access_timestamp) as date,
    COUNT(*) as total_attempts,
    SUM(CASE WHEN access_granted THEN 1 ELSE 0 END) as successful_access,
    SUM(CASE WHEN NOT access_granted THEN 1 ELSE 0 END) as failed_access,
    COUNT(DISTINCT person_name) as unique_persons
FROM access_logs
GROUP BY DATE(access_timestamp)
ORDER BY date DESC;

-- ============================================================
-- FUNCIONES ÚTILES
-- ============================================================

-- Función para obtener el historial de una persona específica
CREATE OR REPLACE FUNCTION get_person_history(p_person_name TEXT, p_limit INT DEFAULT 50)
RETURNS TABLE (
    id BIGINT,
    access_timestamp TIMESTAMPTZ,
    confidence DECIMAL,
    access_granted BOOLEAN,
    event_type TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        al.id,
        al.access_timestamp,
        al.confidence,
        al.access_granted,
        al.event_type
    FROM access_logs al
    WHERE al.person_name = p_person_name
    ORDER BY al.access_timestamp DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Función para limpiar datos antiguos (opcional)
CREATE OR REPLACE FUNCTION cleanup_old_logs(days_to_keep INT DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM access_logs
    WHERE access_timestamp < NOW() - (days_to_keep || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- COMENTARIOS EN LAS TABLAS
-- ============================================================

COMMENT ON TABLE access_logs IS 'Registros de todos los intentos de acceso al sistema de reconocimiento facial';
COMMENT ON TABLE training_sessions IS 'Historial de sesiones de entrenamiento del modelo';

COMMENT ON COLUMN access_logs.person_name IS 'Nombre de la persona reconocida (NULL si es desconocida)';
COMMENT ON COLUMN access_logs.confidence IS 'Nivel de confianza del reconocimiento (más bajo = más confianza)';
COMMENT ON COLUMN access_logs.access_granted IS 'Si el acceso fue exitoso (true) o fallido (false)';
COMMENT ON COLUMN access_logs.event_type IS 'Tipo de evento: successful_access, failed_access, no_face_detected';
COMMENT ON COLUMN access_logs.failure_reason IS 'Razón del fallo si access_granted es false';
COMMENT ON COLUMN access_logs.face_image_base64 IS 'Imagen del rostro codificada en base64';
COMMENT ON COLUMN access_logs.additional_data IS 'Datos adicionales en formato JSON';

-- ============================================================
-- FIN DEL SCRIPT
-- ============================================================

-- Para verificar que todo se creó correctamente:
SELECT 'access_logs' as table_name, COUNT(*) as row_count FROM access_logs
UNION ALL
SELECT 'training_sessions', COUNT(*) FROM training_sessions;