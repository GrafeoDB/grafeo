//! The zone maps a 0.5.x base keeps per block of a column (encoding 3):
//! min, max, null and row counts, inline in the block index. The reader
//! steps past them; the fold does not use them.

/// Steps past a zone map in the inline layout of a v3 block index: the null
/// and row counts, then the min and the max.
///
/// # Errors
///
/// Returns a static-string error on truncation or an unknown value tag.
pub(super) fn skip_inline(data: &[u8], pos: &mut usize) -> Result<(), &'static str> {
    read_inline_u32(data, pos)?;
    read_inline_u32(data, pos)?;
    read_inline_value(data, pos)?;
    read_inline_value(data, pos)?;
    Ok(())
}

/// Steps past one bound of an inline zone map: a tag, then an `i64`, a
/// bool or a length-prefixed string.
fn read_inline_value(data: &[u8], pos: &mut usize) -> Result<(), &'static str> {
    let tag = *data.get(*pos).ok_or("truncated zone map value tag")?;
    *pos += 1;
    match tag {
        0 => Ok(()),
        1 => {
            if *pos + 8 > data.len() {
                return Err("truncated zone map Int64");
            }
            *pos += 8;
            Ok(())
        }
        2 => {
            data.get(*pos).ok_or("truncated zone map Bool")?;
            *pos += 1;
            Ok(())
        }
        3 => {
            let len = read_inline_u32(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err("truncated zone map String");
            }
            std::str::from_utf8(&data[*pos..*pos + len])
                .map_err(|_| "invalid UTF-8 in zone map String")?;
            *pos += len;
            Ok(())
        }
        _ => Err("unknown zone map value tag"),
    }
}

fn read_inline_u32(data: &[u8], pos: &mut usize) -> Result<u32, &'static str> {
    if *pos + 4 > data.len() {
        return Err("truncated zone map u32");
    }
    let bytes: [u8; 4] = data[*pos..*pos + 4].try_into().expect("4 bytes guaranteed");
    *pos += 4;
    Ok(u32::from_le_bytes(bytes))
}
